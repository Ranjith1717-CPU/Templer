"""
Table Processor for Word Documents
Handles complex table operations including row cloning and merged cells.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MergedCell:
    """Represents a merged cell in a table"""
    start_row: int
    start_col: int
    row_span: int
    col_span: int


@dataclass
class TableStructure:
    """Detailed table structure analysis"""
    header_rows: List[int] = field(default_factory=list)    # Row indices that are headers
    data_rows: List[int] = field(default_factory=list)      # Row indices for data (repeatable)
    footer_rows: List[int] = field(default_factory=list)    # Summary/total rows
    merged_cells: List[MergedCell] = field(default_factory=list)
    repeating_pattern: bool = False                          # True if data rows should multiply
    column_count: int = 0
    total_rows: int = 0
    xml_start: int = 0
    xml_end: int = 0
    original_xml: str = ""


class TableProcessor:
    """
    Process complex table operations in Word documents.
    Handles row cloning, merged cells, and data insertion.
    """

    # Patterns for detecting header/footer rows
    HEADER_INDICATORS = [
        r'<w:tblHeader\s*/>',
        r'<w:tblHeader\s+',
        r'<w:b\s*/>',  # Bold text often indicates headers
    ]

    FOOTER_INDICATORS = [
        r'(?:Total|Sum|Grand\s+Total|Subtotal)',
        r'=SUM\(',  # Formula
    ]

    def __init__(self):
        pass

    def analyze_table(self, table_xml: str) -> TableStructure:
        """
        Analyze table structure to identify headers, data rows, and footers.

        Args:
            table_xml: XML content of a single table

        Returns:
            TableStructure with detailed analysis
        """
        structure = TableStructure()
        structure.original_xml = table_xml

        # Find table boundaries
        table_match = re.search(r'<w:tbl[^>]*>(.*?)</w:tbl>', table_xml, re.DOTALL)
        if table_match:
            structure.xml_start = table_match.start()
            structure.xml_end = table_match.end()

        # Extract all rows
        rows = self._extract_rows(table_xml)
        structure.total_rows = len(rows)

        if not rows:
            return structure

        # Analyze each row
        for i, row_xml in enumerate(rows):
            row_type = self._classify_row(row_xml, i, len(rows))

            if row_type == 'header':
                structure.header_rows.append(i)
            elif row_type == 'footer':
                structure.footer_rows.append(i)
            else:
                structure.data_rows.append(i)

        # Detect column count
        structure.column_count = self._get_column_count(rows[0] if rows else "")

        # Detect merged cells
        structure.merged_cells = self._detect_merged_cells(table_xml)

        # Check if data rows have repeating pattern
        structure.repeating_pattern = self._has_repeating_pattern(rows, structure.data_rows)

        return structure

    def detect_repeating_rows(self, table_xml: str) -> Dict:
        """
        Detect which rows in a table should repeat with data.

        Args:
            table_xml: XML content of a table

        Returns:
            Dictionary with header, data, and footer row indices
        """
        structure = self.analyze_table(table_xml)

        return {
            'header': structure.header_rows,
            'data': structure.data_rows,
            'footer': structure.footer_rows
        }

    def clone_row_with_data(self, row_xml: str, data_list: List[Dict]) -> str:
        """
        Clone a template row multiple times with different data.

        Args:
            row_xml: Template row XML
            data_list: List of data dictionaries for each clone

        Returns:
            XML string with all cloned rows
        """
        result_rows = []

        for data in data_list:
            # Clone the row
            new_row = row_xml

            # Find all cells and replace content
            cell_pattern = r'(<w:tc[^>]*>)(.*?)(</w:tc>)'
            cells = list(re.finditer(cell_pattern, new_row, re.DOTALL))

            # Replace cell contents from data (by column index or key)
            for col_idx, cell_match in enumerate(cells):
                cell_start = cell_match.group(1)
                cell_content = cell_match.group(2)
                cell_end = cell_match.group(3)

                # Get value for this cell
                value = None
                if isinstance(data, dict):
                    # Try by column index or column key
                    value = data.get(str(col_idx)) or data.get(f'col_{col_idx}')
                    if value is None and col_idx < len(data):
                        value = list(data.values())[col_idx]
                elif isinstance(data, (list, tuple)) and col_idx < len(data):
                    value = data[col_idx]

                if value is not None:
                    # Replace text content in cell
                    new_content = self._replace_cell_text(cell_content, str(value))
                    new_row = new_row.replace(cell_match.group(0),
                                              cell_start + new_content + cell_end)

            result_rows.append(new_row)

        return ''.join(result_rows)

    def preserve_merged_cells(self, row_xml: str) -> str:
        """
        Ensure merged cell markers are preserved when cloning.

        Args:
            row_xml: Row XML to process

        Returns:
            Row XML with preserved merge markers
        """
        # Find vertical merge markers
        vmerge_pattern = r'<w:vMerge[^>]*/>'

        # For continued merges, we need to keep the vMerge but without restart
        # For first row of merge, we need vMerge w:val="restart"

        # This preserves existing merge structure
        return row_xml

    def insert_table_data(self, table_xml: str, data: List[List[str]],
                          structure: Optional[TableStructure] = None) -> str:
        """
        Insert data into a table, handling row cloning if needed.

        Args:
            table_xml: Original table XML
            data: 2D list of data to insert
            structure: Optional pre-analyzed structure

        Returns:
            Modified table XML
        """
        if structure is None:
            structure = self.analyze_table(table_xml)

        if not structure.data_rows or not data:
            return table_xml

        # Extract all rows
        rows = self._extract_rows(table_xml)

        # Get the template data row
        template_row_idx = structure.data_rows[0]
        template_row = rows[template_row_idx]

        # Clone for each data row
        new_data_rows = []
        for row_data in data:
            # Clone template and fill with data
            new_row = template_row
            cell_pattern = r'(<w:tc[^>]*>)(.*?)(</w:tc>)'
            cells = list(re.finditer(cell_pattern, new_row, re.DOTALL))

            for col_idx, cell_match in enumerate(cells):
                if col_idx < len(row_data):
                    value = row_data[col_idx]
                    cell_start = cell_match.group(1)
                    cell_content = cell_match.group(2)
                    cell_end = cell_match.group(3)

                    new_content = self._replace_cell_text(cell_content, str(value))
                    new_row = new_row.replace(cell_match.group(0),
                                              cell_start + new_content + cell_end)

            new_data_rows.append(new_row)

        # Reconstruct table
        result_rows = []
        for i, row in enumerate(rows):
            if i in structure.header_rows:
                result_rows.append(row)
            elif i == template_row_idx:
                result_rows.extend(new_data_rows)
            elif i not in structure.data_rows:
                result_rows.append(row)
            # Skip other data rows (replaced by clones)

        # Rebuild table XML
        row_xml = ''.join(result_rows)

        # Find table wrapper and replace content
        table_wrapper_pattern = r'(<w:tbl[^>]*>)(.*?)(</w:tbl>)'
        match = re.search(table_wrapper_pattern, table_xml, re.DOTALL)

        if match:
            # Extract table properties
            tbl_content = match.group(2)
            tbl_pr_match = re.search(r'<w:tblPr>.*?</w:tblPr>', tbl_content, re.DOTALL)
            tbl_pr = tbl_pr_match.group(0) if tbl_pr_match else ''

            tbl_grid_match = re.search(r'<w:tblGrid>.*?</w:tblGrid>', tbl_content, re.DOTALL)
            tbl_grid = tbl_grid_match.group(0) if tbl_grid_match else ''

            return match.group(1) + tbl_pr + tbl_grid + row_xml + match.group(3)

        return table_xml

    def _extract_rows(self, table_xml: str) -> List[str]:
        """Extract all row XML elements from table"""
        row_pattern = r'<w:tr[^>]*>.*?</w:tr>'
        return re.findall(row_pattern, table_xml, re.DOTALL)

    def _classify_row(self, row_xml: str, row_index: int, total_rows: int) -> str:
        """
        Classify a row as header, footer, or data.

        Args:
            row_xml: XML of the row
            row_index: Index of the row
            total_rows: Total number of rows

        Returns:
            'header', 'footer', or 'data'
        """
        # Check for explicit header marker
        for pattern in self.HEADER_INDICATORS:
            if re.search(pattern, row_xml):
                return 'header'

        # First row is often header
        if row_index == 0:
            # Check if it has bold text
            if '<w:b/>' in row_xml or '<w:b ' in row_xml:
                return 'header'

        # Check for footer indicators
        row_text = self._extract_row_text(row_xml).lower()
        for pattern in self.FOOTER_INDICATORS:
            if re.search(pattern, row_text, re.IGNORECASE):
                return 'footer'

        # Last row might be footer
        if row_index == total_rows - 1:
            if 'total' in row_text or 'sum' in row_text:
                return 'footer'

        return 'data'

    def _get_column_count(self, row_xml: str) -> int:
        """Count columns in a row, accounting for merged cells"""
        cells = re.findall(r'<w:tc[^>]*>', row_xml)
        col_count = 0

        for cell in cells:
            # Check for gridSpan (horizontal merge)
            span_match = re.search(r'<w:gridSpan w:val="(\d+)"', row_xml)
            if span_match:
                col_count += int(span_match.group(1))
            else:
                col_count += 1

        return col_count

    def _detect_merged_cells(self, table_xml: str) -> List[MergedCell]:
        """Detect all merged cells in the table"""
        merged_cells = []
        rows = self._extract_rows(table_xml)

        # Track vertical merges
        vmerge_start = {}  # col_idx -> start_row

        for row_idx, row_xml in enumerate(rows):
            cells = list(re.finditer(r'<w:tc[^>]*>(.*?)</w:tc>', row_xml, re.DOTALL))

            col_idx = 0
            for cell_match in cells:
                cell_xml = cell_match.group(1)

                # Check for horizontal merge (gridSpan)
                span_match = re.search(r'<w:gridSpan w:val="(\d+)"', cell_xml)
                col_span = int(span_match.group(1)) if span_match else 1

                # Check for vertical merge
                vmerge_match = re.search(r'<w:vMerge[^>]*w:val="([^"]*)"', cell_xml)
                if vmerge_match:
                    if vmerge_match.group(1) == 'restart':
                        # Start of vertical merge
                        vmerge_start[col_idx] = row_idx
                elif '<w:vMerge/>' in cell_xml or '<w:vMerge ' in cell_xml:
                    # Continuation of vertical merge
                    pass
                else:
                    # End of vertical merge
                    if col_idx in vmerge_start:
                        start_row = vmerge_start[col_idx]
                        merged_cells.append(MergedCell(
                            start_row=start_row,
                            start_col=col_idx,
                            row_span=row_idx - start_row,
                            col_span=col_span
                        ))
                        del vmerge_start[col_idx]

                if col_span > 1:
                    merged_cells.append(MergedCell(
                        start_row=row_idx,
                        start_col=col_idx,
                        row_span=1,
                        col_span=col_span
                    ))

                col_idx += col_span

        return merged_cells

    def _has_repeating_pattern(self, rows: List[str], data_row_indices: List[int]) -> bool:
        """Check if data rows have a repeating pattern suggesting multiple entries"""
        if len(data_row_indices) <= 1:
            return True  # Single data row = template for repetition

        # Compare structure of data rows
        if len(data_row_indices) >= 2:
            first_row = rows[data_row_indices[0]]
            second_row = rows[data_row_indices[1]]

            # Count cells
            first_cells = len(re.findall(r'<w:tc[^>]*>', first_row))
            second_cells = len(re.findall(r'<w:tc[^>]*>', second_row))

            return first_cells == second_cells

        return False

    def _extract_row_text(self, row_xml: str) -> str:
        """Extract plain text from a row"""
        text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'
        texts = re.findall(text_pattern, row_xml)
        return ' '.join(texts)

    def _replace_cell_text(self, cell_content: str, new_text: str) -> str:
        """Replace text content in a cell while preserving structure"""
        # Find text elements and replace
        def replace_text(match):
            tag_start = match.group(1)
            tag_end = match.group(3)
            return tag_start + self._escape_xml(new_text) + tag_end

        # Replace first text element, clear others
        first_replaced = False
        result = []

        for line in cell_content.split('<w:t'):
            if not first_replaced and '<w:t' not in line:
                result.append(line)
            elif not first_replaced:
                # Replace content
                t_match = re.match(r'([^>]*>)[^<]*(</w:t>)', '<w:t' + line)
                if t_match:
                    result.append(t_match.group(1) + self._escape_xml(new_text) + t_match.group(2))
                    first_replaced = True
                else:
                    result.append('<w:t' + line)
            else:
                # Clear other text elements
                cleared = re.sub(r'>([^<]*)</w:t>', '></w:t>', '<w:t' + line)
                result.append(cleared)

        return ''.join(result) if result else cell_content

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters"""
        return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))
