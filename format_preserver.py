"""
Format-Preserving Replacer for Word Documents
Handles split runs, formatting preservation, and multi-paragraph replacement.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RunProperties:
    """Text run formatting properties"""
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    font_name: Optional[str] = None
    font_size: Optional[float] = None  # In points
    color: Optional[str] = None
    highlight: Optional[str] = None
    superscript: bool = False
    subscript: bool = False

    def to_xml(self) -> str:
        """Convert properties to Word XML"""
        parts = []

        if self.bold:
            parts.append('<w:b/>')
        if self.italic:
            parts.append('<w:i/>')
        if self.underline:
            parts.append('<w:u w:val="single"/>')
        if self.strike:
            parts.append('<w:strike/>')
        if self.font_name:
            parts.append(f'<w:rFonts w:ascii="{self.font_name}" w:hAnsi="{self.font_name}"/>')
        if self.font_size:
            # Font size in half-points
            half_points = int(self.font_size * 2)
            parts.append(f'<w:sz w:val="{half_points}"/>')
        if self.color:
            parts.append(f'<w:color w:val="{self.color}"/>')
        if self.highlight:
            parts.append(f'<w:highlight w:val="{self.highlight}"/>')
        if self.superscript:
            parts.append('<w:vertAlign w:val="superscript"/>')
        if self.subscript:
            parts.append('<w:vertAlign w:val="subscript"/>')

        if parts:
            return '<w:rPr>' + ''.join(parts) + '</w:rPr>'
        return ''


@dataclass
class SplitRun:
    """Represents text split across multiple runs"""
    combined_text: str
    parts: List[str]
    run_xmls: List[str]
    start_position: int
    paragraph_xml: str


class FormatPreserver:
    """
    Replace content in Word documents while preserving formatting.
    Handles split runs and complex formatting scenarios.
    """

    def __init__(self):
        pass

    def replace_text(self, document_xml: str, old_text: str, new_text: str) -> str:
        """
        Replace text in document while preserving formatting.
        Handles both simple and split run cases.

        Args:
            document_xml: Full document XML
            old_text: Text to replace
            new_text: Replacement text

        Returns:
            Modified document XML
        """
        # First try simple replacement
        if old_text in self._extract_all_text(document_xml):
            # Check if text is split across runs
            if self._is_text_split(document_xml, old_text):
                return self.handle_split_text(document_xml, old_text, new_text)
            else:
                return self._simple_replace(document_xml, old_text, new_text)

        return document_xml

    def handle_split_text(self, document_xml: str, target: str, replacement: str) -> str:
        """
        Handle text that is split across multiple <w:r> elements.

        Args:
            document_xml: Document XML
            target: Target text to find (may be split)
            replacement: Replacement text

        Returns:
            Modified document XML
        """
        # Find paragraphs containing parts of the target
        paragraphs = self._find_paragraphs_with_text(document_xml, target)

        for para_xml in paragraphs:
            # Get all runs in paragraph
            runs = self._extract_runs(para_xml)

            # Concatenate text and track positions
            full_text = ''
            run_boundaries = []  # (start, end, run_xml, run_properties)

            for run_xml in runs:
                text = self._extract_run_text(run_xml)
                props = self._extract_run_properties(run_xml)

                start = len(full_text)
                full_text += text
                run_boundaries.append((start, len(full_text), run_xml, props))

            # Find target in concatenated text
            target_start = full_text.find(target)
            if target_start == -1:
                continue

            target_end = target_start + len(target)

            # Find which runs are affected
            affected_runs = []
            for start, end, run_xml, props in run_boundaries:
                if end > target_start and start < target_end:
                    affected_runs.append({
                        'start': start,
                        'end': end,
                        'xml': run_xml,
                        'properties': props,
                        'target_start': max(start, target_start),
                        'target_end': min(end, target_end)
                    })

            if not affected_runs:
                continue

            # Build replacement - use first run's formatting
            first_props = affected_runs[0]['properties']

            # Create new run with replacement text
            new_run = self._create_run(replacement, first_props)

            # Modify paragraph
            new_para_xml = para_xml

            # Handle text before target in first affected run
            first_run = affected_runs[0]
            if first_run['start'] < target_start:
                prefix_text = full_text[first_run['start']:target_start]
                prefix_run = self._create_run(prefix_text, first_run['properties'])
            else:
                prefix_run = ''

            # Handle text after target in last affected run
            last_run = affected_runs[-1]
            if last_run['end'] > target_end:
                suffix_text = full_text[target_end:last_run['end']]
                suffix_run = self._create_run(suffix_text, last_run['properties'])
            else:
                suffix_run = ''

            # Replace affected runs with new runs
            # First, remove all affected runs
            for run_info in affected_runs:
                new_para_xml = new_para_xml.replace(run_info['xml'], '', 1)

            # Insert replacement at first affected run position
            insert_pos = para_xml.find(affected_runs[0]['xml'])
            replacement_runs = prefix_run + new_run + suffix_run

            # Find position in new_para_xml
            new_para_xml = new_para_xml[:insert_pos] + replacement_runs + new_para_xml[insert_pos:]

            # Update document
            document_xml = document_xml.replace(para_xml, new_para_xml, 1)

        return document_xml

    def preserve_run_properties(self, run_xml: str, new_text: str) -> str:
        """
        Replace text in a run while preserving all formatting.

        Args:
            run_xml: Original run XML
            new_text: New text content

        Returns:
            New run XML with preserved formatting
        """
        props = self._extract_run_properties(run_xml)
        return self._create_run(new_text, props)

    def replace_multiline(self, para_xml: str, replacement: str) -> str:
        """
        Replace paragraph content with multi-line text.
        Creates new paragraphs for each line.

        Args:
            para_xml: Original paragraph XML
            replacement: Multi-line replacement text

        Returns:
            XML for multiple paragraphs
        """
        lines = replacement.split('\n')

        if len(lines) == 1:
            return self._replace_paragraph_text(para_xml, replacement)

        # Clone paragraph for each line
        result_paragraphs = []

        for line in lines:
            new_para = self._replace_paragraph_text(para_xml, line)
            result_paragraphs.append(new_para)

        return ''.join(result_paragraphs)

    def _simple_replace(self, document_xml: str, old_text: str, new_text: str) -> str:
        """Simple text replacement in <w:t> elements"""
        # Escape the new text for XML
        escaped_new = self._escape_xml(new_text)

        # Find and replace within text elements
        pattern = r'(<w:t[^>]*>)([^<]*?)(' + re.escape(old_text) + r')([^<]*?)(</w:t>)'

        def replacer(match):
            return match.group(1) + match.group(2) + escaped_new + match.group(4) + match.group(5)

        return re.sub(pattern, replacer, document_xml)

    def _is_text_split(self, document_xml: str, text: str) -> bool:
        """Check if text is split across multiple runs"""
        # If text appears in a single <w:t> element, it's not split
        pattern = r'<w:t[^>]*>[^<]*' + re.escape(text) + r'[^<]*</w:t>'
        if re.search(pattern, document_xml):
            return False

        # Check if text appears in concatenated paragraph text
        for para_xml in self._find_paragraphs_with_text(document_xml, text):
            runs = self._extract_runs(para_xml)
            combined = ''.join(self._extract_run_text(r) for r in runs)
            if text in combined:
                return True

        return False

    def _find_paragraphs_with_text(self, document_xml: str, text: str) -> List[str]:
        """Find paragraph XML elements that might contain the text"""
        paragraphs = []

        # First, get all paragraphs
        para_pattern = r'<w:p[^>]*>.*?</w:p>'

        for match in re.finditer(para_pattern, document_xml, re.DOTALL):
            para_xml = match.group(0)

            # Extract all text from paragraph
            runs = self._extract_runs(para_xml)
            para_text = ''.join(self._extract_run_text(r) for r in runs)

            # Check if any part of text is in this paragraph
            if text in para_text or any(part in para_text for part in text.split()):
                paragraphs.append(para_xml)

        return paragraphs

    def _extract_runs(self, para_xml: str) -> List[str]:
        """Extract all run elements from a paragraph"""
        run_pattern = r'<w:r[^>]*>.*?</w:r>'
        return re.findall(run_pattern, para_xml, re.DOTALL)

    def _extract_run_text(self, run_xml: str) -> str:
        """Extract text content from a run"""
        text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'
        matches = re.findall(text_pattern, run_xml)
        return ''.join(matches)

    def _extract_run_properties(self, run_xml: str) -> RunProperties:
        """Extract formatting properties from a run"""
        props = RunProperties()

        rpr_match = re.search(r'<w:rPr>(.*?)</w:rPr>', run_xml, re.DOTALL)
        if not rpr_match:
            return props

        rpr_xml = rpr_match.group(1)

        props.bold = '<w:b/>' in rpr_xml or '<w:b ' in rpr_xml
        props.italic = '<w:i/>' in rpr_xml or '<w:i ' in rpr_xml
        props.underline = '<w:u ' in rpr_xml
        props.strike = '<w:strike' in rpr_xml

        # Font name
        font_match = re.search(r'<w:rFonts[^>]*w:ascii="([^"]*)"', rpr_xml)
        if font_match:
            props.font_name = font_match.group(1)

        # Font size
        size_match = re.search(r'<w:sz w:val="(\d+)"', rpr_xml)
        if size_match:
            props.font_size = int(size_match.group(1)) / 2

        # Color
        color_match = re.search(r'<w:color w:val="([^"]*)"', rpr_xml)
        if color_match:
            props.color = color_match.group(1)

        # Highlight
        hl_match = re.search(r'<w:highlight w:val="([^"]*)"', rpr_xml)
        if hl_match:
            props.highlight = hl_match.group(1)

        # Vertical alignment
        if 'vertAlign w:val="superscript"' in rpr_xml:
            props.superscript = True
        if 'vertAlign w:val="subscript"' in rpr_xml:
            props.subscript = True

        return props

    def _create_run(self, text: str, properties: RunProperties) -> str:
        """Create a new run element with text and properties"""
        props_xml = properties.to_xml()
        escaped_text = self._escape_xml(text)

        # Preserve spaces
        if text.startswith(' ') or text.endswith(' '):
            t_attrs = ' xml:space="preserve"'
        else:
            t_attrs = ''

        return f'<w:r>{props_xml}<w:t{t_attrs}>{escaped_text}</w:t></w:r>'

    def _replace_paragraph_text(self, para_xml: str, new_text: str) -> str:
        """Replace all text in a paragraph"""
        # Get first run's properties
        runs = self._extract_runs(para_xml)
        if runs:
            props = self._extract_run_properties(runs[0])
        else:
            props = RunProperties()

        # Create new run
        new_run = self._create_run(new_text, props)

        # Extract paragraph properties
        ppr_match = re.search(r'<w:pPr>.*?</w:pPr>', para_xml, re.DOTALL)
        ppr_xml = ppr_match.group(0) if ppr_match else ''

        return f'<w:p>{ppr_xml}{new_run}</w:p>'

    def _extract_all_text(self, document_xml: str) -> str:
        """Extract all text from document"""
        text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'
        matches = re.findall(text_pattern, document_xml)
        return ' '.join(matches)

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters"""
        return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


def create_format_preserving_replacer():
    """Factory function for FormatPreserver"""
    return FormatPreserver()
