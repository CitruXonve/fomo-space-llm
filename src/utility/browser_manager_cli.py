"""Generic BrowserManager CLI entry (no LinkedIn dependencies)."""

from __future__ import annotations

from src.utility.browser_cli_common import (
    add_browser_args,
    add_generic_subparsers,
    async_main_generic,
    browser_echo,
    browser_from_args,
    build_generic_parser,
    dispatch_generic_command,
    emit_error,
    emit_error_payload,
    emit_ok,
    execute_generic_step,
    main_generic,
    parse_viewport,
    read_js_code,
    run_cli,
    run_command,
    truncate_text,
    with_browser,
)

# Backward-compatible aliases for tests and internal imports.
_parse_viewport = parse_viewport
_add_browser_args = add_browser_args
_browser_from_args = browser_from_args
_browser_echo = browser_echo
_emit_ok = emit_ok
_emit_error = emit_error
_emit_error_payload = emit_error_payload
_truncate_text = truncate_text
_read_js_code = read_js_code
_execute_step = execute_generic_step
_run_command = run_command
_with_browser = with_browser
_build_parser = build_generic_parser

execute_step = execute_generic_step
build_parser = build_generic_parser


def main() -> int:
    return main_generic()


if __name__ == "__main__":
    raise SystemExit(main())
