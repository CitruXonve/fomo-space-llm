import json
import unittest
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
            "/ByteByteGo _ Technical Interview Prep - 02 Scale From Zero To Millions Of Users.pdf"
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
              len(sections[0]), sections[0][:50])

        with open(settings.EXPORT_DIRECTORY + "/sections_case1.json", "w") as f:
            json.dump(sections, f, indent=4)

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
              len(sections[-1]), sections[-1][:50])


if __name__ == "__main__":
    unittest.main()
