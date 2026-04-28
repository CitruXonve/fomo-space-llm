"""LangChain agent that drives ``BrowserManager`` to operate on LinkedIn.

The agent is a thin orchestration layer around :class:`BrowserManager` and a
``langchain.agents.create_agent`` instance configured with ChatAnthropic and a
small set of browser-control tools.

Run-time flow:

1. Start the browser (persistent profile by default so saved login is reused).
2. Pre-flight authentication check: navigate to the LinkedIn feed; if the user
   is not signed in, navigate to the sign-in page and wait up to
   ``login_timeout_s`` seconds for the user to complete login.
3. If ``headless=True`` was set but auth is missing, automatically relaunch
   the browser non-headless so the user can interact with the sign-in form.
4. Invoke the LangChain agent with the LinkedIn job-search task.
5. Aggregate token usage from the returned messages and return everything
   along with the elapsed wall-clock time.

The class itself emits zero stdout output. Callers wire ``on_progress`` to
their own renderer (e.g. a CLI spinner).
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.messages import AnyMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from src.config.settings import settings
from src.utility.browser_manager import BrowserManager
from src.utility.linkedin_feed_parser import activity_id_from_url, parse_feed_posts
from src.utility.spinner import Spinner

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """
You are a LinkedIn job-search intelligence agent. Your goal is to surface relevant hiring
opportunities and draft personalized outreach messages for the user.

IMPORTANT CONTEXT:
- The browser is already authenticated to LinkedIn -- DO NOT attempt to sign in or navigate to login pages.
- You are running inside an automated browser controlled by the available tools.

CONSTRAINTS:
- Collect at most {max_posts} qualifying posts.
- Only consider posts published within the last {recency_hours} hours.

GUARDRAILS:
- Never paste raw HTML into your responses; bulk feed ingestion is done server-side via ``collect_raw_feed_posts`` (BeautifulSoup), which returns compact JSON only.
- Never paste base64 screenshot payloads back into your responses; treat them as opaque evidence only.
- Never invent URLs -- only record URLs you have observed from tool output (e.g. ``collect_raw_feed_posts``) or visible text from targeted extraction.
- Prefer ``collect_raw_feed_posts`` over ``extract_visible_text`` for loading many feed posts (lower tokens). Use the available tools to navigate or scroll only as needed outside that flow.
- In ``execute_javascript``, ``document.querySelector`` / ``querySelectorAll`` only accept **browser CSS** (as in DevTools). Do not use Playwright-only selectors such as ``:has-text()``, ``:nth-match()``, or ``text=`` — they will throw. Prefer ``aria-label``, role-based queries, XPath via ``document.evaluate``, or walking the DOM.

OUTPUT REQUIREMENTS:
- Save the final result as JSON via the `save_output_to_file` tool.
- Filename pattern: `linkedin_posts_<timestamp>.json` where <timestamp> is a unix epoch or ISO-8601 string.
- Each entry must include: company_url, job_listing_url (nullable), author_profile_url, post_url.
""".strip()


DEFAULT_LINKEDIN_TASK_TEMPLATE = """
Execute the following steps IN ORDER using the available tools:

1. Navigate to the LinkedIn feed page (https://www.linkedin.com/feed/), then click "sort by" and select "recent".
2. Call ``collect_raw_feed_posts`` to gather up to {max_posts} raw posts (parses HTML in Python; scroll as needed via that tool — do not paste HTML). Keep posts published within the last {recency_hours} hours based on ``relative_time`` / snippet text when present.
3. For each raw post that looks like a hiring announcement, record:
   a. The company's LinkedIn URL.
   b. The job listing URL (if present, otherwise null).
   c. The author's profile URL.
   d. The post URL.
4. Output the collected entries in JSON format using the `save_output_to_file` tool with the filename `linkedin_posts_<timestamp>.json`.
""".strip()


def _format_duration(total_seconds: float) -> str:
    """Format a duration as 'Xm Y.YYs' (always shows minutes, even when 0)."""
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}m {seconds:.2f}s"


async def _spin_until(
    stop: asyncio.Event,
    current_stage: list[str],
    spinner: Spinner,
) -> None:
    """Drive ``spinner.spin`` until ``stop`` is set.

    ``current_stage`` is a 1-element list used as a mutable slot for the
    in-flight stage label so callers can update it atomically by reassigning
    index 0.
    """
    while not stop.is_set():
        spinner.spin(type_str=current_stage[0])
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            pass


class LinkedInWebAgent:
    """LangChain agent that operates a real LinkedIn session via Playwright."""

    def __init__(
        self,
        user_data_dir: str | Path = ".browser_profile",
        profile_directory: str | None = None,
        headless: bool = False,
        model: str | None = None,
        export_dir: str | None = None,
        max_posts: int = 5,
        recency_hours: int = 48,
        login_timeout_s: int = 120,
    ) -> None:
        self.user_data_dir = Path(user_data_dir)
        self.profile_directory = profile_directory
        self.headless = headless
        self.max_posts = max_posts
        self.recency_hours = recency_hours
        self.login_timeout_s = login_timeout_s

        self._model_name = model or settings.CLAUDE_MODEL
        self._export_dir = export_dir or settings.EXPORT_DIRECTORY

        self.browser = BrowserManager(
            headless=self.headless,
            user_data_dir=self.user_data_dir,
            profile_directory=self.profile_directory,
        )

        self.agent = self._build_agent()

    async def __aenter__(self) -> "LinkedInWebAgent":
        await self.browser.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.browser.stop()

    async def run(
        self,
        task: str | None = None,
        timeout_seconds: int = 300,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict:
        """Execute the LinkedIn job-search task once and return a summary dict."""
        emit = self._make_progress_emitter(on_progress)
        start = time.perf_counter()

        emit("starting browser")
        async with self.browser:
            await self._ensure_authenticated(emit)

            emit("running agent")
            human_task = task or DEFAULT_LINKEDIN_TASK_TEMPLATE.format(
                max_posts=self.max_posts,
                recency_hours=self.recency_hours,
            )

            try:
                response = await asyncio.wait_for(
                    self.agent.ainvoke(
                        {"messages": [HumanMessage(content=human_task)]},
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "LinkedIn agent run exceeded timeout of %ds", timeout_seconds
                )
                raise

            emit("finalizing")
            messages = response.get("messages", [])
            usage = self._aggregate_token_usage(messages)

        elapsed_seconds = time.perf_counter() - start
        elapsed_pretty = _format_duration(elapsed_seconds)

        logger.info(
            "LinkedInWebAgent run complete in %s -- input=%d, output=%d, total=%d, ai_turns=%d, tool_calls=%d",
            elapsed_pretty,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
            usage["ai_message_count"],
            usage["tool_call_count"],
        )

        return {
            "messages": messages,
            "token_usage": usage,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_pretty": elapsed_pretty,
        }

    async def _ensure_authenticated(
        self,
        emit: Callable[[str], None],
    ) -> None:
        emit("checking authentication")
        await self.browser.navigate("https://www.linkedin.com/feed/")

        if await self._is_authenticated():
            logger.info("LinkedIn session already authenticated")
            return

        if self.headless:
            logger.warning(
                "LinkedIn not authenticated while headless=True; "
                "relaunching browser non-headless so the user can sign in."
            )
            await self.browser.stop()
            self.browser.headless = False
            self.headless = False
            await self.browser.start()

        emit("waiting for sign-in")
        await self.browser.navigate("https://www.linkedin.com/login")

        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.login_timeout_s
        attempts = 0
        while loop.time() < deadline:
            attempts += 1
            is_authed = await self._is_authenticated()
            if is_authed:
                logger.info("LinkedIn sign-in completed by user")
                return
            await asyncio.sleep(2.0)

        raise RuntimeError(
            f"LinkedIn sign-in not completed within {self.login_timeout_s}s"
        )

    async def _is_authenticated(self) -> bool:
        current_url = await self.browser.get_current_url()
        blocked_markers = ("/login", "/uas/login", "/checkpoint", "/authwall")
        matched_marker = next((m for m in blocked_markers if m in current_url), None)
        if matched_marker is not None:
            return False
        try:
            probe = await self.browser.execute_javascript(
                """(() => {
                    const hasGlobalNav = Boolean(
                        document.querySelector('nav.global-nav, [data-test-global-nav], .global-nav')
                    );
                    const hasSearchInput = Boolean(
                        document.querySelector('input[placeholder*="Search"], input[aria-label*="Search"]')
                    );
                    const hasNetworkLink = Boolean(
                        document.querySelector('a[href*="/mynetwork/"], a[href*="/jobs/"], a[href*="/messaging/"]')
                    );
                    const hasSignInForm = Boolean(
                        document.querySelector('form[action*="login"], input[name="session_key"], input[name="session_password"]')
                    );
                    const hasJoinLink = Boolean(
                        document.querySelector('a[href*="signup"], a[data-tracking-control-name*="join"]')
                    );
                    return {
                        hasGlobalNav,
                        hasSearchInput,
                        hasNetworkLink,
                        hasSignInForm,
                        hasJoinLink,
                    };
                })()"""
            )
        except Exception as exc:
            logger.debug("Auth DOM probe failed: %s", exc)
            return False
        if not isinstance(probe, dict):
            probe = {}
        has_global_nav = bool(probe.get("hasGlobalNav"))
        has_search_input = bool(probe.get("hasSearchInput"))
        has_network_link = bool(probe.get("hasNetworkLink"))
        has_sign_in_form = bool(probe.get("hasSignInForm"))
        has_join_link = bool(probe.get("hasJoinLink"))

        has_logged_in_hint = has_global_nav or has_search_input or has_network_link
        has_signed_out_hint = has_sign_in_form or has_join_link
        on_feed = "/feed/" in current_url

        is_authenticated = has_logged_in_hint or (on_feed and not has_signed_out_hint)
        return is_authenticated

    def _build_agent(self):
        return create_agent(
            model=ChatAnthropic(
                model=self._model_name,
                temperature=settings.CLAUDE_TEMPERATURE,
                max_tokens=settings.CLAUDE_MAX_TOKENS,
            ),
            tools=self._build_tools(),
            middleware=[self._build_prompt_middleware()],
        )

    def _build_prompt_middleware(self):
        self_ref = self

        @dynamic_prompt
        def system_prompt(_request: ModelRequest) -> str:
            return SYSTEM_PROMPT_TEMPLATE.format(
                max_posts=self_ref.max_posts,
                recency_hours=self_ref.recency_hours,
            )

        return system_prompt

    def _build_tools(self) -> list:
        browser = self.browser
        export_dir = self._export_dir
        agent_max_posts = self.max_posts

        @tool
        async def navigate(url: str) -> str:
            """Navigate the browser to a URL. Returns the resolved final URL."""
            await browser.navigate(url)
            return f"navigated to {await browser.get_current_url()}"

        @tool
        async def get_current_url() -> str:
            """Return the URL currently displayed in the browser."""
            return await browser.get_current_url()

        @tool
        async def get_page_title() -> str:
            """Return the document.title of the current page."""
            return await browser.get_page_title()

        @tool
        async def extract_visible_text(max_chars: int = 8000) -> str:
            """Return ``document.body.innerText`` for the current page, truncated to ``max_chars``."""
            text = await browser.extract_visible_text()
            returned_text = text
            if max_chars > 0 and len(text) > max_chars:
                returned_text = (
                    text[:max_chars]
                    + f"\n... [truncated, {len(text) - max_chars} more chars]"
                )
            return returned_text

        @tool
        async def execute_javascript(code: str) -> str:
            """Run JS in the page (same as the browser console). Results are JSON-serialized.

            ``querySelector`` / ``querySelectorAll`` only support standard CSS selectors understood
            by Chromium — not Playwright locator syntax (e.g. ``:has-text('x')``). For text matching
            use XPath (``document.evaluate``), ``TreeWalker``, or filter elements after a broad query.
            """
            result = await browser.execute_javascript(code)
            try:
                return json.dumps(result, default=str)
            except (TypeError, ValueError):
                return str(result)

        @tool
        async def take_screenshot() -> str:
            """Verify a viewport PNG can be captured; image bytes are not returned (token limit).

            Raw base64 must not be placed in tool output — it spans hundreds of thousands of tokens.
            Use ``extract_visible_text`` or DOM tools for page content.
            """
            b64 = await browser.get_screenshot()
            if not b64:
                return "screenshot failed (empty capture)"
            b64_len = len(b64)
            approx_bytes = (b64_len * 3) // 4
            return (
                f"Screenshot OK (PNG, {b64_len} base64 chars, ~{approx_bytes // 1024} KiB). "
                "Image data omitted from model context; use extract_visible_text or execute_javascript for content."
            )

        @tool
        async def scroll(pixels: int = 800) -> str:
            """Scroll the page vertically by ``pixels`` (positive = down)."""
            await browser.scroll(pixels)
            return f"scrolled {pixels}px"

        @tool
        async def collect_raw_feed_posts(
            scroll_rounds: int = 3,
            max_posts: int | None = None,
            html_selector: str | None = None,
        ) -> str:
            """Collect LinkedIn feed posts by parsing the current page HTML server-side.

            Raw HTML is never returned to the model — only a compact JSON list of dicts with
            ``post_url``, ``author_profile_url``, ``text_snippet``, and ``relative_time``.
            Performs ``scroll_rounds`` scroll-load cycles after the initial capture (capped).

            Cap ``max_posts`` is enforced against the agent run limit."""
            target = max_posts if max_posts is not None else agent_max_posts
            target = max(1, min(int(target), agent_max_posts))
            sr = max(0, min(int(scroll_rounds), 10))

            merged: list[dict[str, str]] = []
            seen: set[str] = set()

            def _key(row: dict[str, str]) -> str:
                uid = activity_id_from_url(row.get("post_url", "") or "")
                return uid or (row.get("post_url") or "")[:512]

            for i in range(sr + 1):
                html = await browser.get_page_html(html_selector)
                if not html:
                    logger.warning("collect_raw_feed_posts: empty HTML")
                    break
                batch = parse_feed_posts(html)
                for row in batch:
                    k = _key(row)
                    if k in seen:
                        continue
                    seen.add(k)
                    merged.append(dict(row))
                    if len(merged) >= target:
                        break
                if len(merged) >= target:
                    break
                if i < sr:
                    await browser.scroll(800)
                    await asyncio.sleep(0.75)

            return json.dumps(merged[:target], ensure_ascii=False)

        @tool
        async def wait_for(
            selector: str | None = None,
            timeout_ms: int = 10_000,
        ) -> str:
            """Wait for ``selector`` to appear, or sleep ``timeout_ms`` ms if no selector is given."""
            await browser.wait_for(selector=selector, timeout_ms=timeout_ms)
            return "waited"

        @tool
        async def save_output_to_file(filename: str, content: str) -> bool:
            """Save ``content`` into ``<export_dir>/<filename>``. Returns True on success."""
            try:
                os.makedirs(export_dir, exist_ok=True)
                path = os.path.join(export_dir, filename)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                logger.info("Saved output to %s", path)
                return True
            except Exception as exc:
                logger.error("Failed to save output to %s: %s", filename, exc)
                return False

        return [
            navigate,
            get_current_url,
            get_page_title,
            extract_visible_text,
            execute_javascript,
            take_screenshot,
            scroll,
            collect_raw_feed_posts,
            wait_for,
            save_output_to_file,
        ]

    @staticmethod
    def _aggregate_token_usage(messages: list[AnyMessage]) -> dict:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        input_token_details: dict[str, int] = {}
        ai_message_count = 0
        tool_call_count = 0

        for msg in messages:
            usage = getattr(msg, "usage_metadata", None)
            if usage is None:
                continue
            ai_message_count += 1
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)

            details = usage.get("input_token_details") or {}
            for key, value in details.items():
                input_token_details[key] = input_token_details.get(key, 0) + int(
                    value or 0
                )

            tool_calls = getattr(msg, "tool_calls", None) or []
            tool_call_count += len(tool_calls)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_token_details": input_token_details,
            "ai_message_count": ai_message_count,
            "tool_call_count": tool_call_count,
        }

    @staticmethod
    def _make_progress_emitter(
        on_progress: Callable[[str], None] | None,
    ) -> Callable[[str], None]:
        if on_progress is None:
            return lambda _msg: None

        def emit(msg: str) -> None:
            try:
                on_progress(msg)
            except Exception as exc:
                logger.warning("on_progress callback raised: %s", exc)

        return emit


__all__ = [
    "LinkedInWebAgent",
    "DEFAULT_LINKEDIN_TASK_TEMPLATE",
    "SYSTEM_PROMPT_TEMPLATE",
    "_format_duration",
    "_spin_until",
]
