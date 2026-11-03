"""LinkedIn-specific BrowserManager CLI (auth + feed collect on top of generic CLI)."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.utility.browser_cli_common import (
    add_browser_args,
    add_generic_subparsers,
    browser_echo,
    browser_from_args,
    dispatch_generic_command,
    emit_error,
    emit_error_payload,
    emit_ok,
    execute_generic_step,
    run_cli,
    run_command,
    with_browser,
)
from src.utility.browser_manager import BrowserManager
from src.utility.linkedin_auth import is_linkedin_authenticated, wait_for_linkedin_sign_in
from src.utility.linkedin_feed_collect import (
    collect_hiring_posts_from_page,
    collect_raw_feed_posts_from_page,
    resolve_post_urls_via_send_button,
)

LINKEDIN_OPS = frozenset({"feed-posts", "hiring-posts", "check-auth", "sign-in"})


async def execute_linkedin_step(bm: BrowserManager, step: dict[str, Any]) -> Any:
    op = step.get("op")
    if not op or not isinstance(op, str):
        raise ValueError("each step must have a string 'op' field")

    if op == "feed-posts":
        rows = await collect_raw_feed_posts_from_page(
            bm,
            max_posts=int(step.get("max_posts", 5)),
            scroll_rounds=int(step.get("scroll_rounds", 3)),
            html_selector=step.get("html_selector"),
        )
        return {"posts": rows}

    if op == "hiring-posts":
        rows = await collect_hiring_posts_from_page(
            bm,
            max_posts=int(step.get("max_posts", 10)),
            scroll_px=int(step.get("scroll_px", 800)),
            wait_ms=int(step.get("wait_ms", 2000)),
            max_scrolls=int(step.get("max_scrolls", 25)),
            plateau_limit=int(step.get("plateau_limit", 3)),
        )
        rows = await resolve_post_urls_via_send_button(
            bm,
            rows,
            max_attempt_per_post=int(step.get("max_attempt_per_post", 8)),
        )
        return {"posts": rows}

    if op == "check-auth":
        await bm.navigate("https://www.linkedin.com/feed/")
        authed = await is_linkedin_authenticated(bm)
        return {"authenticated": authed, "current_url": await bm.get_current_url()}

    if op == "sign-in":
        timeout_s = int(step.get("timeout_s", 120))
        authed = await wait_for_linkedin_sign_in(bm, timeout_s=timeout_s)
        return {"authenticated": authed, "timeout_s": timeout_s}

    raise ValueError(f"unknown op: {op!r}")


async def execute_combined_step(bm: BrowserManager, step: dict[str, Any]) -> Any:
    op = step.get("op")
    if isinstance(op, str) and op in LINKEDIN_OPS:
        return await execute_linkedin_step(bm, step)
    return await execute_generic_step(bm, step)


def add_linkedin_subparsers(sub: argparse._SubParsersAction) -> None:
    p_feed = sub.add_parser(
        "feed-posts",
        help="Collect LinkedIn feed posts as compact JSON",
    )
    p_feed.add_argument("--max-posts", type=int, default=5)
    p_feed.add_argument("--scroll-rounds", type=int, default=3)
    p_feed.add_argument("--html-selector", default=None)

    p_hiring = sub.add_parser(
        "hiring-posts",
        help="Collect LinkedIn hiring-announcement posts as structured JSON",
    )
    p_hiring.add_argument("--max-posts", type=int, default=10)
    p_hiring.add_argument("--scroll-px", type=int, default=800)
    p_hiring.add_argument("--wait-ms", type=int, default=2000)
    p_hiring.add_argument("--max-scrolls", type=int, default=25)
    p_hiring.add_argument("--plateau-limit", type=int, default=7)

    sub.add_parser(
        "check-auth",
        help="Check LinkedIn authentication for current profile",
    )
    p_sign = sub.add_parser(
        "sign-in",
        help="Open LinkedIn login and wait for manual sign-in",
    )
    p_sign.add_argument("--timeout-s", type=int, default=120)


def build_linkedin_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LinkedIn BrowserManager CLI (JSON stdout, generic + LinkedIn commands).",
    )
    add_browser_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    add_generic_subparsers(sub)
    add_linkedin_subparsers(sub)
    return parser


async def dispatch_linkedin_command(ns: argparse.Namespace, bm: BrowserManager) -> int | None:
    """Run a LinkedIn-only subcommand. Returns exit code if handled, else ``None``."""
    cmd = ns.command

    if cmd == "feed-posts":

        async def _feed(b: BrowserManager) -> dict[str, Any]:
            rows = await collect_raw_feed_posts_from_page(
                b,
                max_posts=ns.max_posts,
                scroll_rounds=ns.scroll_rounds,
                html_selector=ns.html_selector,
            )
            return {"posts": rows}

        data = await with_browser(bm, _feed)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "hiring-posts":

        async def _hiring(b: BrowserManager) -> dict[str, Any]:
            rows = await collect_hiring_posts_from_page(
                b,
                max_posts=ns.max_posts,
                scroll_px=ns.scroll_px,
                wait_ms=ns.wait_ms,
                max_scrolls=ns.max_scrolls,
                plateau_limit=ns.plateau_limit,
            )
            return {"posts": await resolve_post_urls_via_send_button(b, rows)}

        data = await with_browser(bm, _hiring)
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "check-auth":

        async def _check_auth(b: BrowserManager) -> dict[str, Any]:
            await b.navigate("https://www.linkedin.com/feed/")
            authed = await is_linkedin_authenticated(b)
            return {"authenticated": authed, "current_url": await b.get_current_url()}

        data = await with_browser(bm, _check_auth)
        if not data["authenticated"]:
            emit_error_payload(
                cmd,
                {
                    "error": "linkedin_not_authenticated",
                    "action": "invoke_skill:linkedin-sign-in",
                    "current_url": data["current_url"],
                },
                exit_code=1,
            )
        emit_ok(cmd, data, browser=bm)
        return 0

    if cmd == "sign-in":

        async def _sign_in(b: BrowserManager) -> dict[str, Any]:
            authed = await wait_for_linkedin_sign_in(b, timeout_s=ns.timeout_s)
            return {
                "authenticated": authed,
                "timeout_s": ns.timeout_s,
                "current_url": await b.get_current_url(),
            }

        data = await with_browser(bm, _sign_in)
        if not data["authenticated"]:
            emit_error_payload(
                cmd,
                {
                    "error": "linkedin_sign_in_timeout",
                    "timeout_s": ns.timeout_s,
                    "current_url": data["current_url"],
                },
                exit_code=1,
            )
        emit_ok(cmd, data, browser=bm)
        return 0

    return None


async def async_main_linkedin(ns: argparse.Namespace) -> int:
    try:
        bm = browser_from_args(ns)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    cmd = ns.command
    try:
        if cmd == "run":
            try:
                steps = json.loads(ns.steps)
            except json.JSONDecodeError as exc:
                emit_error(cmd, f"invalid --steps JSON: {exc}")
            if not isinstance(steps, list):
                emit_error(cmd, "--steps must be a JSON array")
            echo = browser_echo(bm)
            try:
                results = await run_command(
                    bm, steps, execute_step=execute_combined_step
                )
            except Exception as exc:
                emit_error(cmd, str(exc))
            emit_ok(
                cmd,
                {"steps": results},
                browser_echo_payload=echo,
                session_mode="multi_step",
            )
            return 0

        code = await dispatch_generic_command(ns, bm)
        if code is not None:
            return code
        code = await dispatch_linkedin_command(ns, bm)
        if code is not None:
            return code
        emit_error("?", f"unknown command: {cmd}")
    except SystemExit:
        raise
    except Exception as exc:
        emit_error(cmd, str(exc))
    return 1


def main() -> int:
    return run_cli(build_parser=build_linkedin_parser, async_main=async_main_linkedin)


__all__ = [
    "LINKEDIN_OPS",
    "build_linkedin_parser",
    "execute_combined_step",
    "execute_linkedin_step",
    "main",
]
