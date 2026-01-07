"""
Enhanced parsing system to handle multiple file formats (Markdown, PDF, TXT, etc.)
"""

from abc import ABC, abstractmethod
from pathlib import Path
import re
import logging
from typing import TypeAlias
from src.type.enums import ContentFormat


logger = logging.getLogger(__name__)

Section: TypeAlias = tuple[str, str]  # (heading, content)

# Abstract base parser


class FileParser(ABC):
    """Base class for file format parsers"""

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file"""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> list[Section]:
        """
        Parse file into sections.
        Returns: list of Section tuples
        """
        pass

    @abstractmethod
    def get_format(self) -> ContentFormat:
        """Get the content format of the file"""
        pass


class MarkdownParser(FileParser):
    """Parser for Markdown (.md) files"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.md'

    def parse(self, file_path: Path) -> list[Section]:
        """Parse markdown into sections based on headers"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self._parse_markdown_sections(content)

    def get_format(self) -> ContentFormat:
        return ContentFormat.MARKDOWN

    def _parse_markdown_sections(self, content: str) -> list[Section]:
        """Parse markdown content by headers"""
        header_pattern = r'^(#{1,6})\s+(.+)$'
        sections = []
        current_heading = "Introduction"
        current_content = []

        lines = content.split('\n')

        for line in lines:
            match = re.match(header_pattern, line, re.MULTILINE)

            if match:
                # Save previous section
                if current_content:
                    sections.append((
                        current_heading,
                        '\n'.join(current_content).strip()
                    ))

                # Start new section
                current_heading = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append((
                current_heading,
                '\n'.join(current_content).strip()
            ))

        return sections


class PDFParser(FileParser):
    """Parser for PDF (.pdf) files"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'

    def get_format(self) -> ContentFormat:
        return ContentFormat.PDF

    def parse(self, file_path: Path) -> list[Section]:
        """Parse PDF into sections"""
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(str(file_path))
        documents = loader.load()

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        import json
        # Aggregate sections by page label
        from collections import defaultdict
        sections_dict = defaultdict(list)

        for doc in chunks:
            section = doc.page_content.strip()
            if not section:
                continue
            print("Page metadata:", json.dumps(doc.metadata, indent=4))

            page_label = f"Page Label {doc.metadata.get('page', 'Unknown')}"
            text_lines = [text.lstrip()
                          for text in section.split('\n') if text.lstrip()]
            sections_dict[page_label].extend(text_lines)

        # Convert dict to list of tuples, maintaining insertion order
        sections = [(page_label, lines)
                    for page_label, lines in sections_dict.items()]

        return sections


class TextParser(FileParser):
    """Parser for plain text (.txt) files"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.txt'

    def get_format(self) -> ContentFormat:
        return ContentFormat.TXT

    def parse(self, file_path: Path) -> list[Section]:
        """Parse plain text into sections"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to detect sections using common patterns
        sections = self._detect_text_sections(content)

        if not sections:
            # Fallback: treat entire file as one section
            sections = [(file_path.stem, content.strip())]

        return sections

    def _detect_text_sections(self, content: str) -> list[Section]:
        """Detect sections in plain text files"""
        sections = []

        lines = content.split('\n')
        current_heading = None
        current_content = []

        for i, line in enumerate(lines):
            # Check for Pattern 1 - underlined headings: Lines with "===" or "---" underlines
            if i > 0 and re.match(r'^[=\-]{3,}$', line.strip()):
                # Previous line is the heading
                if current_heading and current_content:
                    sections.append(
                        (current_heading, '\n'.join(current_content[:-1]).strip()))

                current_heading = current_content[-1] if current_content else "Section"
                current_content = []
                continue

            # Check for Pattern 2 - numbered sections: Lines starting with numbers (1. 2. etc.)
            numbered_match = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
            if numbered_match and len(line.strip()) < 100:
                if current_heading and current_content:
                    sections.append(
                        (current_heading, '\n'.join(current_content).strip()))

                current_heading = numbered_match.group(2).strip()
                current_content = []
                continue

            current_content.append(line)

        # Add final section
        if current_heading and current_content:
            sections.append(
                (current_heading, '\n'.join(current_content).strip()))

        return sections


# Parser factory
class ParserFactory:
    """Factory to create appropriate parser based on file type"""

    def __init__(self):
        # Register parsers in order of preference
        self.parsers: list[FileParser] = [
            MarkdownParser(),
            PDFParser(),
            TextParser(),
        ]

    def get_parser_and_format(self, file_path: Path) -> tuple[FileParser, ContentFormat] | None:
        """Get appropriate parser for the file"""
        for parser in self.parsers:
            try:
                if parser.can_parse(file_path):
                    return parser, parser.get_format()
            except ImportError as e:
                logger.debug(
                    f"Parser {parser.__class__.__name__} not available: {e}")
                continue

        logger.warning(f"No parser found for file: {file_path}")
        return None

    def parse_file(self, file_path: Path) -> list[Section]:
        """Parse file using appropriate parser"""
        parser, _ = self.get_parser_and_format(file_path)

        if parser is None:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Parsing {file_path} with {parser.__class__.__name__}")
        return parser.parse(file_path)
