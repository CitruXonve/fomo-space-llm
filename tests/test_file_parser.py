import json
import unittest
from statistics import mean

from src.config.settings import settings
from src.service.file_parser import ParserFactory
from pathlib import Path

from src.type.enums import ContentFormat


class TestFileParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser_factory = ParserFactory()

    def test_pdf_parser(self):
        """
        Use browser to print the content from:
        - https://bytebytego.com/courses/system-design-interview/scale-from-zero-to-millions-of-users
        - Save as a PDF file to the `.knowledge_sources` directory.
        """
        pdf_path = settings.KB_DIRECTORY + \
            "/ByteByteGo _ Technical Interview Prep - clear.pdf"
        self.assertIsNotNone(self.parser_factory.get_parser_and_format(
            Path(pdf_path)))
        parser, format = self.parser_factory.get_parser_and_format(
            Path(pdf_path))
        self.assertIsNotNone(parser)
        self.assertIsNotNone(format)
        self.assertEqual(format, ContentFormat.PDF)
        sections = self.parser_factory.parse_file(Path(pdf_path))
        self.assertGreater(len(sections), 0)
        print(f"Sections from {format.value}:",
              json.dumps([section[1] for section in sections], indent=4))

        with open(settings.EXPORT_DIRECTORY + "/sections_case1.json", "w") as f:
            json.dump(sections, f, indent=4)

    def test_pdf_parser_no_prechunk(self):
        """Semantic path: fewer, larger sections with richer headings when possible."""
        pdf_path = settings.KB_DIRECTORY + \
            "/ByteByteGo _ Technical Interview Prep - clear.pdf"
        path = Path(pdf_path)
        if not path.is_file():
            self.skipTest(f"PDF fixture not present: {pdf_path}")

        prechunked = self.parser_factory.parse_file(
            path, args={"prechunk": True})
        full_sections = self.parser_factory.parse_file(
            path, args={"prechunk": False})

        self.assertGreater(len(prechunked), 0)
        self.assertGreater(len(full_sections), 0)
        self.assertLess(len(full_sections), len(prechunked))

        avg_prechunk = mean(len(s[1]) for s in prechunked)
        avg_full = mean(len(s[1]) for s in full_sections)
        self.assertGreater(avg_full, avg_prechunk)

        md5_only = sum(
            1 for h, _ in full_sections if "-" in h and len(h) > 20)
        self.assertLess(md5_only, len(full_sections))

    def test_markdown_parser(self):
        """
        Download the raw markdown file from:
        - https://github.com/CitruXonve/devblog/blob/master/source/_posts/k8s-entry.md
        - Save the markdown file to the `.knowledge_sources` directory.
        """
        markdown_path = settings.KB_DIRECTORY + "/k8s-entry.md"
        self.assertIsNotNone(self.parser_factory.get_parser_and_format(
            Path(markdown_path)))
        parser, format = self.parser_factory.get_parser_and_format(
            Path(markdown_path))
        self.assertIsNotNone(parser)
        self.assertIsNotNone(format)
        self.assertEqual(format, ContentFormat.MARKDOWN)
        sections = self.parser_factory.parse_file(Path(markdown_path))
        self.assertGreater(len(sections), 0)
        print(f"Sections from {format.value}:",
              json.dumps([section[1] for section in sections], indent=4))


if __name__ == "__main__":
    unittest.main()
