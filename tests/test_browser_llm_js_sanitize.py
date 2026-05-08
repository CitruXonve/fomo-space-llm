"""Tests for LLM JavaScript sanitization used by ``BrowserManager.execute_javascript``."""

import unittest

from src.utility.browser_manager import (
    _normalize_llm_javascript,
    _strip_llm_typescript_noise,
    _wrap_for_page_evaluate,
)


class TestStripLlmTypescriptNoise(unittest.TestCase):
    def test_const_primitive_annotation(self) -> None:
        src = 'const title: string = document.title\nreturn title'
        out = _strip_llm_typescript_noise(src)
        self.assertNotIn(": string", out)
        self.assertIn("const title =", out)

    def test_arrow_single_param(self) -> None:
        src = "(el: HTMLElement) => el.outerHTML"
        out = _strip_llm_typescript_noise(src)
        self.assertEqual(out, "(el) => el.outerHTML")

    def test_wrap_produces_valid_function_body(self) -> None:
        wrapped = _wrap_for_page_evaluate(
            "const x: string = 'ok'\nreturn x"
        )
        self.assertIn("new Function", wrapped)
        self.assertNotIn(": string", wrapped)

    def test_normalize_strips_fence_and_types(self) -> None:
        raw = "```js\nconst n: number = 1\nreturn n\n```"
        out = _normalize_llm_javascript(raw)
        self.assertNotIn(": number", out)
        self.assertIn("const n =", out)


if __name__ == "__main__":
    unittest.main()
