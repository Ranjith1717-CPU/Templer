"""
Document Structure Parser for Word Documents
Deep OOXML parsing with heading hierarchy, tables, and structure detection.
"""

import re
import io
import zipfile
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class ElementType(Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class TextRun:
    """Represents a text run within a paragraph"""
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    highlight_color: Optional[str] = None
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    color: Optional[str] = None
    xml_start: int = 0
    xml_end: int = 0


@dataclass
class Paragraph:
    """Represents a paragraph with all runs and properties"""
    runs: List[TextRun] = field(default_factory=list)
    style: Optional[str] = None
    heading_level: Optional[int] = None
    alignment: Optional[str] = None
    xml_position: int = 0
    full_text: str = ""

    def get_text(self) -> str:
        return "".join(run.text for run in self.runs)


@dataclass
class TableCell:
    """Represents a table cell"""
    paragraphs: List[Paragraph] = field(default_factory=list)
    row_span: int = 1
    col_span: int = 1
    width: Optional[int] = None

    def get_text(self) -> str:
        return " ".join(p.get_text() for p in self.paragraphs)


@dataclass
class TableRow:
    """Represents a table row"""
    cells: List[TableCell] = field(default_factory=list)
    is_header: bool = False
    height: Optional[int] = None


@dataclass
class Table:
    """Represents a complete table structure"""
    rows: List[TableRow] = field(default_factory=list)
    xml_position: int = 0
    style: Optional[str] = None

    @property
    def column_count(self) -> int:
        return max((len(row.cells) for row in self.rows), default=0)

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class Heading:
    """Represents a document heading"""
    text: str
    level: int
    position: int
    paragraph: Paragraph


@dataclass
class MergedCell:
    """Represents a merged cell in a table"""
    start_row: int
    start_col: int
    row_span: int
    col_span: int


@dataclass
class DocumentStructure:
    """Complete document structure"""
    paragraphs: List[Paragraph] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    headings: List[Heading] = field(default_factory=list)
    document_xml: str = ""
    styles_xml: str = ""
    numbering_xml: str = ""
    raw_bytes: bytes = b""

    def get_full_text(self) -> str:
        """Get all text from the document"""
        texts = []
        for para in self.paragraphs:
            texts.append(para.get_text())
        for table in self.tables:
            for row in table.rows:
                for cell in row.cells:
                    texts.append(cell.get_text())
        return "\n".join(texts)

    def get_heading_hierarchy(self) -> Dict[str, Any]:
        """Get heading hierarchy as nested dict"""
        hierarchy = {"title": None, "sections": []}
        current_section = None
        current_subsection = None

        for heading in sorted(self.headings, key=lambda h: h.position):
            if heading.level == 1:
                if hierarchy["title"] is None:
                    hierarchy["title"] = heading.text
                else:
                    current_section = {"title": heading.text, "subsections": []}
                    hierarchy["sections"].append(current_section)
                    current_subsection = None
            elif heading.level == 2 and current_section:
                current_subsection = {"title": heading.text, "items": []}
                current_section["subsections"].append(current_subsection)
            elif heading.level >= 3 and current_subsection:
                current_subsection["items"].append(heading.text)

        return hierarchy


class DocumentParser:
    """
    Deep OOXML parser for Word documents.
    Provides comprehensive document structure analysis.
    """

    # Namespace prefixes for Word XML
    NS = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    }

    # Heading style patterns
    HEADING_PATTERNS = [
        (r'Heading\s*(\d+)', lambda m: int(m.group(1))),
        (r'Title', lambda m: 1),
        (r'Subtitle', lambda m: 2),
    ]

    def __init__(self):
        self.document_xml = ""
        self.styles_xml = ""
        self.numbering_xml = ""
        self.style_map = {}

    def parse_document(self, file_bytes: bytes) -> DocumentStructure:
        """
        Parse a Word document from bytes.

        Args:
            file_bytes: Raw bytes of the .docx file

        Returns:
            DocumentStructure with all parsed elements
        """
        structure = DocumentStructure(raw_bytes=file_bytes)

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                # Extract core XML files
                if 'word/document.xml' in zf.namelist():
                    structure.document_xml = zf.read('word/document.xml').decode('utf-8')
                    self.document_xml = structure.document_xml

                if 'word/styles.xml' in zf.namelist():
                    structure.styles_xml = zf.read('word/styles.xml').decode('utf-8')
                    self.styles_xml = structure.styles_xml
                    self._parse_styles()

                if 'word/numbering.xml' in zf.namelist():
                    structure.numbering_xml = zf.read('word/numbering.xml').decode('utf-8')
                    self.numbering_xml = structure.numbering_xml
        except Exception as e:
            raise ValueError(f"Failed to read document: {e}")

        # Parse document structure
        structure.paragraphs = self._extract_paragraphs()
        structure.tables = self._extract_tables()
        structure.headings = self._extract_headings(structure.paragraphs)

        return structure

    def _parse_styles(self):
        """Parse styles.xml to build style map"""
        if not self.styles_xml:
            return

        # Extract style definitions
        style_pattern = r'<w:style[^>]*w:styleId="([^"]*)"[^>]*>.*?</w:style>'
        for match in re.finditer(style_pattern, self.styles_xml, re.DOTALL):
            style_id = match.group(1)
            style_xml = match.group(0)

            # Check if it's a heading style
            for pattern, level_func in self.HEADING_PATTERNS:
                if re.search(pattern, style_id, re.IGNORECASE):
                    self.style_map[style_id] = {
                        'is_heading': True,
                        'level': level_func(re.search(pattern, style_id, re.IGNORECASE))
                    }
                    break
            else:
                self.style_map[style_id] = {'is_heading': False}

    def _extract_paragraphs(self) -> List[Paragraph]:
        """Extract all paragraphs from document"""
        paragraphs = []

        # Find all paragraph elements
        para_pattern = r'<w:p[^>]*>(.*?)</w:p>'
        for pos, match in enumerate(re.finditer(para_pattern, self.document_xml, re.DOTALL)):
            para_xml = match.group(1)
            para = Paragraph(xml_position=match.start())

            # Extract paragraph style
            style_match = re.search(r'<w:pStyle w:val="([^"]*)"', para_xml)
            if style_match:
                para.style = style_match.group(1)
                style_info = self.style_map.get(para.style, {})
                if style_info.get('is_heading'):
                    para.heading_level = style_info.get('level')

            # Extract text runs
            para.runs = self._extract_runs(para_xml)
            para.full_text = para.get_text()

            paragraphs.append(para)

        return paragraphs

    def _extract_runs(self, para_xml: str) -> List[TextRun]:
        """Extract text runs from paragraph XML"""
        runs = []

        # Find all run elements
        run_pattern = r'<w:r[^>]*>(.*?)</w:r>'
        for match in re.finditer(run_pattern, para_xml, re.DOTALL):
            run_xml = match.group(1)

            # Extract text
            text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'
            text_matches = re.findall(text_pattern, run_xml)
            text = ''.join(text_matches)

            if not text:
                continue

            run = TextRun(
                text=text,
                xml_start=match.start(),
                xml_end=match.end()
            )

            # Extract run properties
            rpr_match = re.search(r'<w:rPr>(.*?)</w:rPr>', run_xml, re.DOTALL)
            if rpr_match:
                rpr_xml = rpr_match.group(1)
                run.bold = '<w:b/>' in rpr_xml or '<w:b ' in rpr_xml
                run.italic = '<w:i/>' in rpr_xml or '<w:i ' in rpr_xml
                run.underline = '<w:u ' in rpr_xml

                # Check for highlight
                hl_match = re.search(r'<w:highlight w:val="([^"]*)"', rpr_xml)
                if hl_match:
                    run.highlight_color = hl_match.group(1)

                # Check for shading (background color)
                shd_match = re.search(r'<w:shd[^>]*w:fill="([^"]*)"', rpr_xml)
                if shd_match and shd_match.group(1) not in ('auto', 'FFFFFF', 'ffffff'):
                    run.highlight_color = run.highlight_color or shd_match.group(1)

                # Font size
                sz_match = re.search(r'<w:sz w:val="(\d+)"', rpr_xml)
                if sz_match:
                    run.font_size = int(sz_match.group(1)) / 2  # Half-points to points

                # Font name
                font_match = re.search(r'<w:rFonts[^>]*w:ascii="([^"]*)"', rpr_xml)
                if font_match:
                    run.font_name = font_match.group(1)

                # Text color
                color_match = re.search(r'<w:color w:val="([^"]*)"', rpr_xml)
                if color_match:
                    run.color = color_match.group(1)

            runs.append(run)

        return runs

    def _extract_tables(self) -> List[Table]:
        """Extract all tables from document"""
        tables = []

        # Find all table elements
        table_pattern = r'<w:tbl[^>]*>(.*?)</w:tbl>'
        for match in re.finditer(table_pattern, self.document_xml, re.DOTALL):
            table_xml = match.group(1)
            table = Table(xml_position=match.start())

            # Extract table style
            style_match = re.search(r'<w:tblStyle w:val="([^"]*)"', table_xml)
            if style_match:
                table.style = style_match.group(1)

            # Extract rows
            row_pattern = r'<w:tr[^>]*>(.*?)</w:tr>'
            for row_idx, row_match in enumerate(re.finditer(row_pattern, table_xml, re.DOTALL)):
                row_xml = row_match.group(1)
                row = TableRow(is_header=(row_idx == 0))

                # Check if row is marked as header
                if '<w:tblHeader/>' in row_xml or '<w:tblHeader ' in row_xml:
                    row.is_header = True

                # Extract cells
                cell_pattern = r'<w:tc[^>]*>(.*?)</w:tc>'
                for cell_match in re.finditer(cell_pattern, row_xml, re.DOTALL):
                    cell_xml = cell_match.group(1)
                    cell = TableCell()

                    # Check for merged cells
                    vmerge = re.search(r'<w:vMerge[^>]*w:val="([^"]*)"', cell_xml)
                    hmerge = re.search(r'<w:gridSpan w:val="(\d+)"', cell_xml)

                    if hmerge:
                        cell.col_span = int(hmerge.group(1))

                    # Extract paragraphs in cell
                    cell_para_pattern = r'<w:p[^>]*>(.*?)</w:p>'
                    for para_match in re.finditer(cell_para_pattern, cell_xml, re.DOTALL):
                        para = Paragraph()
                        para.runs = self._extract_runs(para_match.group(1))
                        para.full_text = para.get_text()
                        cell.paragraphs.append(para)

                    row.cells.append(cell)

                table.rows.append(row)

            tables.append(table)

        return tables

    def _extract_headings(self, paragraphs: List[Paragraph]) -> List[Heading]:
        """Extract headings from paragraphs"""
        headings = []

        for para in paragraphs:
            if para.heading_level:
                headings.append(Heading(
                    text=para.get_text(),
                    level=para.heading_level,
                    position=para.xml_position,
                    paragraph=para
                ))

        return headings

    def extract_heading_hierarchy(self) -> List[Dict]:
        """Get heading hierarchy as nested structure"""
        structure = self.parse_document(self.document_xml.encode() if isinstance(self.document_xml, str) else b'')
        return structure.get_heading_hierarchy()

    def find_split_runs(self) -> List[Dict]:
        """
        Find text that is split across multiple runs.
        This happens when Word applies different formatting mid-word.

        Returns:
            List of split run groups with their positions
        """
        split_runs = []

        # Find paragraphs where text appears split
        para_pattern = r'<w:p[^>]*>(.*?)</w:p>'
        for para_match in re.finditer(para_pattern, self.document_xml, re.DOTALL):
            para_xml = para_match.group(1)

            # Get all text elements
            text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'
            texts = re.findall(text_pattern, para_xml)

            # Check for split words (text elements that don't end with space)
            for i, text in enumerate(texts[:-1]):
                if text and not text.endswith(' ') and texts[i+1] and not texts[i+1].startswith(' '):
                    # This text is likely split
                    combined = text + texts[i+1]
                    split_runs.append({
                        'combined_text': combined,
                        'parts': [text, texts[i+1]],
                        'paragraph_position': para_match.start()
                    })

        return split_runs

    def get_context_for_position(self, position: int, chars_before: int = 200, chars_after: int = 50) -> str:
        """Get surrounding text context for a position in the document"""
        # Extract all text with positions
        full_text = self._get_text_with_positions()

        # Find the text segment around the position
        text_only = ''.join(t['text'] for t in full_text)

        # Map XML position to text position (approximate)
        current_pos = 0
        text_pos = 0
        for segment in full_text:
            if current_pos >= position:
                break
            current_pos = segment['end']
            text_pos += len(segment['text'])

        start = max(0, text_pos - chars_before)
        end = min(len(text_only), text_pos + chars_after)

        return text_only[start:end]

    def _get_text_with_positions(self) -> List[Dict]:
        """Get all text segments with their XML positions"""
        segments = []
        text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'

        for match in re.finditer(text_pattern, self.document_xml):
            segments.append({
                'text': match.group(1),
                'start': match.start(),
                'end': match.end()
            })

        return segments
