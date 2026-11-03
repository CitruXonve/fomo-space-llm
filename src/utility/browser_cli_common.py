"""Shared BrowserManager CLI helpers and generic commands (no LinkedIn deps)."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from src.utility.browser_manager import BrowserManager

_PREVIEW_B64_CHARS = 120

ExecuteStepFn = Callable[[BrowserManager, dict[str, Any]], Awaitable[Any]]


def parse_viewport(raw: str) -> tuple[int, int]:
    if "x" not in raw.lower():
        raise ValueError(f"viewport must be WIDTHxHEIGHT, got: {raw!r}")
    w_s, h_s = raw.lower().split("x", 1)
    w, h = int(w_s), int(h_s)
    if w < 1 or h < 1:
        raise ValueError(f"viewport dimensions must be positive: {raw!r}")
    return (w, h)


def add_browser_args(parser: argparse.ArgumentParser) -> None:
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="Run Chromium headless (default when neither headless flag is set)",
    )
    headless.add_argument(
        "--no-headless",
        action="store_true",
        default=None,
        help="Run Chromium with a visible window",
    )
    parser.add_argument(
        "--viewport",
        default="1280x720",
        metavar="WxH",
        help="Browser viewport (default: %(default)s)",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        dest="user_agent",
        help="Custom User-Agent (default: BrowserManager built-in)",
    )
    parser.add_argument(
        "--user-data-dir",
        default=None,
        dest="user_data_dir",
        metavar="PATH",
        help="Persistent Chrome profile directory",
    )
    parser.add_argument(
        "--profile-directory",
        default=None,
        dest="profile_directory",
        metavar="NAME",
        help="Chrome profile name (e.g. Default); use with --user-data-dir",
    )


def browser_from_args(ns: argparse.Namespace) -> BrowserManager:
    if ns.headless is True:
        headless = True
    elif ns.no_headless is True:
        headless = False
    else:
        headless = True

    viewport = parse_viewport(ns.viewport)

    return BrowserManager(
        headless=headless,
        viewport=viewport,
        user_agent=ns.user_agent,
        user_data_dir=ns.user_data_dir,
        profile_directory=ns.profile_directory,
    )


def browser_echo(bm: BrowserManager) -> dict[str, Any]:
    return {
        "headless": bm.headless,
        "viewport": list(bm.viewport),
        "user_data_dir": str(bm.user_data_dir) if bm.user_data_dir else None,
        "profile_directory": bm.profile_directory,
        "user_agent": bm.user_agent,
    }


def emit_ok(
    command: str,
    data: Any,
    *,
    browser: BrowserManager | None = None,
    browser_echo_payload: dict[str, Any] | None = None,
    session_mode: str = "ephemeral",
) -> None:
    payload: dict[str, Any] = {
        "ok": True,
        "command": command,
        "session_mode": session_mode,
        "data": data,
    }
    if browser_echo_payload is not None:
        payload["browser"] = browser_echo_payload
    elif browser is not None:
        payload["browser"] = browser_echo(browser)
    print(json.dumps(payload, ensure_ascii=False, default=str))


def emit_error(command: str, error: str, *, exit_code: int = 1) -> None:
    print(
        json.dumps({"ok": False, "command": command, "error": error}),
        file=sys.stderr,
    )
    raise SystemExit(exit_code)


def emit_error_payload(command: str, payload: dict[str, Any], *, exit_code: int = 1) -> None:
    body = {"ok": False, "command": command}
    body.update(payload)
    print(json.dumps(body, ensure_ascii=False, default=str), file=sys.stderr)
    raise SystemExit(exit_code)


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"


def read_js_code(ns: argparse.Namespace) -> str:
    if ns.file:
        return Path(ns.file).read_text(encoding="utf-8")
    if ns.code is not None:
        return ns.code
    emit_error("js", "provide CODE positional argument or --file PATH")


async def execute_generic_step(bm: BrowserManager, step: dict[str, Any]) -> Any:
    op = step.get("op")
    if not op or not isinstance(op, str):
        raise ValueError("each step must have a string 'op' field")

    if op == "navigate":
        url = step.get("url")
        if not url:
            raise ValueError("navigate step requires 'url'")
        await bm.navigate(str(url))
        return {"url": await bm.get_current_url()}

    if op == "url":
        return {"url": await bm.get_current_url()}

    if op == "title":
        return {"title": await bm.get_page_title()}

    if op == "text":
        text = await bm.extract_visible_text()
        max_chars = int(step.get("max_chars", 0))
        if max_chars > 0:
            text = truncate_text(text, max_chars)
        return {"text": text}

    if op == "html":
        selector = step.get("selector")
        html = await bm.get_page_html(selector)
        max_chars = int(step.get("max_chars", 0))
        if max_chars > 0:
            html = truncate_text(html, max_chars)
        return {"html": html}

    if op == "js":
        code = step.get("code")
        if not code and step.get("file"):
            code = Path(str(step["file"])).read_text(encoding="utf-8")
        if not code:
            raise ValueError("js step requires 'code' or 'file'")
        result = await bm.execute_javascript(str(code))
        return {"result": result}

    if op == "screenshot":
        b64 = await bm.get_screenshot()
        out_path = step.get("out")
        if out_path:
            Path(out_path).write_bytes(base64.b64decode(b64))
            return {"path": str(out_path), "base64_length": len(b64)}
        preview = b64[:_PREVIEW_B64_CHARS] if b64 else ""
        return {
            "base64_preview": preview,
            "base64_length": len(b64),
        }

    if op == "scroll":
        pixels = int(step.get("pixels", 800))
        await bm.scroll(pixels)
        return {"scrolled": pixels}

    if op == "wait":
        selector = step.get("selector")
        timeout_ms = int(step.get("timeout_ms", 10_000))
        await bm.wait_for(selector=selector, timeout_ms=timeout_ms)
        return {"waited": True, "selector": selector}

    if op == "scroll-until":
        selector = step.get("selector")
        if not selector:
            raise ValueError("scroll-until step requires 'selector'")
        min_count = int(step.get("min_count", 1))
        scroll_px = int(step.get("pixels", 800))
        wait_ms = int(step.get("wait_ms", 2000))
        max_rounds = int(step.get("max_rounds", 20))
        rounds = 0
        while rounds < max_rounds:
            count = await bm.execute_javascript(
                f"return document.querySelectorAll({selector!r}).length"
            )
            if int(count or 0) >= min_count:
                return {"rounds": rounds, "count": count, "selector": selector}
            await bm.scroll(scroll_px)
            await bm.wait_for(timeout_ms=wait_ms)
            rounds += 1
        count = await bm.execute_javascript(
            f"return document.querySelectorAll({selector!r}).length"
        )
        return {"rounds": rounds, "count": count, "selector": selector, "exhausted": True}

    raise ValueError(f"unknown op: {op!r}")


async def run_command(
    bm: BrowserManager,
    steps: list[dict[str, Any]],
    *,
    execute_step: ExecuteStepFn,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    async with bm:
        for step in steps:
            op = step.get("op", "?")
            try:
                data = await execute_step(bm, step)
                results.append({"op": op, "ok": True, "data": data})
            except Exception as exc:
                results.append({"op": op, "ok": False, "error": str(exc)})
                raise
    return results


async def with_browser(bm: BrowserManager, fn: Any) -> Any:
    async with bm:
        return await fn(bm)


def add_generic_subparsers(sub: argparse._SubParsersAction) -> None:
    p_nav = sub.add_parser("navigate", help="Navigate to URL")
    p_nav.add_argument("url")

    sub.add_parser("url", help="Print current URL")
    sub.add_parser("title", help="Print page title")

    p_text = sub.add_parser("text", help="Extract visible body text")
    p_text.add_argument("--max-chars", type=int, default=0)

    p_html = sub.add_parser("html", help="Capture page HTML")
    p_html.add_argument("--selector", default=None)
    p_html.add_argument("--max-chars", type=int, default=0)

    p_js = sub.add_parser("js", help="Execute JavaScript on the page")
    p_js.add_argument("code", nargs="?", default=None)
    p_js.add_argument("--file", default=None, help="Read JS from file")

    p_shot = sub.add_parser("screenshot", help="Capture viewport PNG")
    p_shot.add_argument("--out", default=None, help="Write PNG to path")

    p_scroll = sub.add_parser("scroll", help="Scroll vertically")
    p_scroll.add_argument("pixels", type=int)

    p_wait = sub.add_parser("wait", help="Wait for selector or timeout")
    p_wait.add_argument("--selector", default=None)
    p_wait.add_argument("--timeout-ms", type=int, default=10_000)

    p_run = sub.add_parser("run", help="Run multiple steps in one session")
    p_run.add_argument(
        "--steps",
        required=True,
        help="JSON array of step objects",
    )


def build_generic_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Playwright BrowserManager CLI (JSON stdout, generic commands).",
    )
    add_browser_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    add_generic_subparsers(sub)
    return parser


async def dispatch_generic_command(ns: argparse.Namespace, bm: BrowserManager) -> int | None:
    """Run a generic subcommand. Returns exit code if handled, else ``None``."""
    cmd = ns.command

    if cmd == "navigate":

        async def _nav(b: BrowserManager) -> dict[str, Any]:
            await b.navigate(ns.url)
            return {"url": await b.get_current_url()}

        data = await with_browser(bm, _nav)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "url":

        async def _url(b: BrowserManager) -> dict[str, Any]:
            return {"url": await b.get_current_url()}

        data = await with_browser(bm, _url)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "title":

        async def _title(b: BrowserManager) -> dict[str, Any]:
            title = await b.get_page_title()
            url = await b.get_current_url()
            payload: dict[str, Any] = {"title": title, "url": url}
            if not title:
                payload["hint"] = (
                    "empty_title_usually_means_a_new_browser_session_without_navigate; "
                    "chain steps with run --steps instead of separate CLI invocations"
                )
            return payload

        data = await with_browser(bm, _title)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "text":

        async def _text(b: BrowserManager) -> dict[str, Any]:
            text = await b.extract_visible_text()
            if ns.max_chars > 0:
                text = truncate_text(text, ns.max_chars)
            return {"text": text}

        data = await with_browser(bm, _text)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "html":

        async def _html(b: BrowserManager) -> dict[str, Any]:
            html = await b.get_page_html(ns.selector)
            if ns.max_chars > 0:
                html = truncate_text(html, ns.max_chars)
            return {"html": html}

        data = await with_browser(bm, _html)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "js":
        code = read_js_code(ns)

        async def _js(b: BrowserManager) -> dict[str, Any]:
            result = await b.execute_javascript(code)
            return {"result": result}

        data = await with_browser(bm, _js)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "screenshot":

        async def _shot(b: BrowserManager) -> dict[str, Any]:
            b64 = await b.get_screenshot()
            if not b64:
                return {"error": "empty capture"}
            if ns.out:
                Path(ns.out).write_bytes(base64.b64decode(b64))
                return {"path": ns.out, "base64_length": len(b64)}
            return {
                "base64_preview": b64[:_PREVIEW_B64_CHARS],
                "base64_length": len(b64),
            }

        data = await with_browser(bm, _shot)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "scroll":

        async def _scroll(b: BrowserManager) -> dict[str, Any]:
            await b.scroll(ns.pixels)
            return {"scrolled": ns.pixels}

        data = await with_browser(bm, _scroll)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "wait":

        async def _wait(b: BrowserManager) -> dict[str, Any]:
            await b.wait_for(selector=ns.selector, timeout_ms=ns.timeout_ms)
            return {"waited": True, "selector": ns.selector}

        data = await with_browser(bm, _wait)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "run":
        try:
            steps = json.loads(ns.steps)
        except json.JSONDecodeError as exc:
            emit_error(cmd, f"invalid --steps JSON: {exc}")
        if not isinstance(steps, list):
            emit_error(cmd, "--steps must be a JSON array")
        echo = browser_echo(bm)
        try:
            results = await run_command(bm, steps, execute_step=execute_generic_step)
        except Exception as exc:
            emit_error(cmd, str(exc))
        emit_ok(
            cmd,
            {"steps": results},
            browser_echo_payload=echo,
            session_mode="multi_step",
        )
        return 0

    return None


async def async_main_generic(ns: argparse.Namespace) -> int:
    try:
        bm = browser_from_args(ns)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    cmd = ns.command
    try:
        code = await dispatch_generic_command(ns, bm)
        if code is not None:
            return code
        emit_error("?", f"unknown command: {cmd}")
    except SystemExit:
        raise
    except Exception as exc:
        emit_error(cmd, str(exc))
    return 1


def run_cli(
    *,
    build_parser: Callable[[], argparse.ArgumentParser],
    async_main: Callable[[argparse.Namespace], Awaitable[int]],
) -> int:
    try:
        parser = build_parser()
        ns = parser.parse_args()
        try:
            parse_viewport(ns.viewport)
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 2
        return asyncio.run(async_main(ns))
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1


def main_generic() -> int:
    return run_cli(build_parser=build_generic_parser, async_main=async_main_generic)
