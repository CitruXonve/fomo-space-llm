"""
Enhanced parsing system to handle multiple file formats (Markdown, PDF, TXT, etc.)
"""

from abc import ABC, abstractmethod
from pathlib import Path
import re
import logging
from typing import TypeAlias
from src.config.settings import settings
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
    def parse(self, file_path: Path, args: dict = {}) -> list[Section]:
        """
        Parse file into sections.
        Returns: list of Section tuples
        """
        pass

    @abstractmethod
    def get_format(self) -> ContentFormat:
        """Get the content format of the file"""
        pass


def parse_markdown_to_sections(content: str) -> list[Section]:
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


class MarkdownParser(FileParser):
    """Parser for Markdown (.md) files"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.md'

    def parse(self, file_path: Path, args: dict = {}) -> list[Section]:
        """Parse markdown into sections based on headers"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self._parse_markdown_sections(content)

    def get_format(self) -> ContentFormat:
        return ContentFormat.MARKDOWN

    def _parse_markdown_sections(self, content: str) -> list[Section]:
        """Parse markdown content by headers"""
        return parse_markdown_to_sections(content)


class PDFParser(FileParser):
    """Parser for PDF (.pdf) files"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'

    def get_format(self) -> ContentFormat:
        return ContentFormat.PDF

    def _is_header_footer_line(self, line: str) -> bool:
        """Detect if a line is likely a header or footer based on patterns"""
        line_stripped = line.strip()

        # Skip empty lines
        if not line_stripped:
            return True

        # Pattern 1: Date/time stamps (e.g., "12/30/25, 3:02 PM")
        date_time_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M'
        if re.search(date_time_pattern, line_stripped, re.IGNORECASE):
            return True

        # Pattern 2: URLs (http:// or https://)
        if re.match(r'https?://', line_stripped):
            return True

        # Pattern 3: Page numbers (e.g., "1/28", "2/28")
        page_number_pattern = r'^\d+/\d+$'
        if re.match(page_number_pattern, line_stripped):
            return True

        # Pattern 4: Short lines that are likely site names/headers (< 60 chars with | separator)
        if len(line_stripped) < 60 and '|' in line_stripped:
            return True

        return False

    def _filter_repeated_lines(self, sections_dict: dict, threshold: float = 0.3) -> dict:
        """Filter out lines that appear in many pages (likely headers/footers)

        Args:
            sections_dict: Dictionary mapping page labels to lists of text lines
            threshold: If a line appears in more than this fraction of pages, remove it
        """
        from collections import Counter

        # Count occurrences of each line across all pages
        line_counts = Counter()
        total_pages = len(sections_dict)

        for lines in sections_dict.values():
            unique_lines = set(line.strip() for line in lines)
            line_counts.update(unique_lines)

        # Identify lines that appear too frequently (likely headers/footers)
        repeated_lines = {
            line for line, count in line_counts.items()
            if count / total_pages > threshold and len(line.strip()) < 100
        }

        # Filter out repeated lines from each page
        filtered_dict = {}
        for page_label, lines in sections_dict.items():
            filtered_lines = [
                line for line in lines
                if line.strip() not in repeated_lines
            ]
            if filtered_lines:  # Only keep pages with content
                filtered_dict[page_label] = filtered_lines

        return filtered_dict

    def parse(self, file_path: Path, args: dict = {}) -> list[Section]:
        """Parse PDF into sections"""
        from hashlib import md5
        md5_hash = md5(file_path.read_bytes()).hexdigest()

        import pymupdf.layout  # avoid falling back to legacy mode that ignores use_ocr=False
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(str(
            file_path), headers=False, footers=False, page_chunks=True, write_images=False, show_progress=True, use_ocr=False)

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.get(
                'chunk_size', settings.EMBEDDING_MODEL_CHUNK_SIZE),
            chunk_overlap=args.get(
                'chunk_overlap', settings.EMBEDDING_MODEL_CHUNK_OVERLAP),
            separators=["\n\n", "\n", " ", ""]
        )

        sections = []
        for page in md_text:
            parsed_sections = parse_markdown_to_sections(page["text"])
            for parsed_section in parsed_sections:
                text_chunks = text_splitter.split_text(parsed_section[1])
                for text_chunk in text_chunks:
                    if text_chunk.strip():
                        sections.append(text_chunk)

        return [(f"{md5_hash}-{section_index}", section_content)
                for section_index, section_content in enumerate(sections)]


class TextParser(FileParser):
    """Parser for plain text (.txt) files"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.txt'

    def get_format(self) -> ContentFormat:
        return ContentFormat.TXT

    def parse(self, file_path: Path, args: dict = {}) -> list[Section]:
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
            # Check for underlined headings
            if i > 0 and re.match(r'^[=\-]{3,}$', line.strip()):
                # Previous line is the heading
                if current_heading and current_content:
                    sections.append(
                        (current_heading, '\n'.join(current_content[:-1]).strip()))

                current_heading = current_content[-1] if current_content else "Section"
                current_content = []
                continue

            # Check for numbered sections
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

    def parse_file(self, file_path: Path, args: dict = {}) -> list[Section]:
        """Parse file using appropriate parser"""
        parser, _ = self.get_parser_and_format(file_path)

        if parser is None:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Parsing {file_path} with {parser.__class__.__name__}")
        return parser.parse(file_path, args)

if __name__ == "__main__":
    parser_factory = ParserFactory()
    for file in Path(settings.KB_DIRECTORY).rglob("*"):
        if file.is_file():
            parser, format = parser_factory.get_parser_and_format(file)
            if parser is not None:
                print(f"Find {parser.__class__.__name__} compatible with {format.value} type for file: {file.relative_to(settings.KB_DIRECTORY)}")
                logger.info(f"Find {parser.__class__.__name__} compatible with {format.value} type for file: {file.relative_to(settings.KB_DIRECTORY)}")
