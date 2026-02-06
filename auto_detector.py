"""
Auto-Detection Engine for Template Insertion Points
Detects insertion points WITHOUT manual highlighting.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from .patterns import PatternMatcher, PATTERNS, PatternMatch, extract_context_label
from .document_parser import DocumentParser, DocumentStructure, Paragraph, TextRun


@dataclass
class InsertionPoint:
    """Represents a detected insertion point in a template"""
    text: str                          # The text/value to be replaced
    context: str                       # Context before the value (e.g., "Client Name:")
    field_type: str                    # Inferred field type (currency, date, name, etc.)
    detection_method: str              # How it was detected (pattern, heuristic, context, highlight)
    confidence: float                  # Confidence score 0-1
    xml_position: int                  # Position in document XML
    paragraph_index: int = -1          # Index in paragraph list
    run_index: int = -1                # Index in run list
    is_highlighted: bool = False       # Whether it has highlighting
    highlight_color: Optional[str] = None

    def __hash__(self):
        return hash((self.text, self.xml_position))

    def __eq__(self, other):
        if not isinstance(other, InsertionPoint):
            return False
        return self.text == other.text and abs(self.xml_position - other.xml_position) < 50


@dataclass
class DetectionResult:
    """Result of auto-detection including all insertion points"""
    insertion_points: List[InsertionPoint] = field(default_factory=list)
    detection_stats: Dict[str, int] = field(default_factory=dict)
    overall_confidence: float = 0.0

    def get_by_type(self, field_type: str) -> List[InsertionPoint]:
        """Get insertion points filtered by type"""
        return [ip for ip in self.insertion_points if ip.field_type == field_type]

    def get_high_confidence(self, threshold: float = 0.7) -> List[InsertionPoint]:
        """Get only high confidence insertion points"""
        return [ip for ip in self.insertion_points if ip.confidence >= threshold]


class AutoDetector:
    """
    Auto-detection engine for finding template insertion points.
    Combines multiple detection strategies for maximum coverage.
    """

    # Context keywords that suggest a value follows
    CONTEXT_KEYWORDS = {
        'name': ['client', 'name', 'customer', 'applicant', 'advisor', 'adviser', 'mr', 'mrs', 'ms', 'dr'],
        'currency': ['amount', 'value', 'total', 'sum', 'cost', 'fee', 'charge', 'price', 'balance'],
        'date': ['date', 'dated', 'on', 'from', 'until', 'expires', 'expiry', 'effective'],
        'percentage': ['rate', 'return', 'growth', 'yield', 'interest', 'percentage', 'percent'],
        'reference': ['reference', 'ref', 'number', 'no', 'id', 'account', 'policy'],
        'provider': ['provider', 'platform', 'wrapper', 'company', 'institution'],
        'address': ['address', 'located', 'lives', 'resident'],
        'email': ['email', 'e-mail', 'contact'],
        'phone': ['phone', 'telephone', 'mobile', 'contact', 'call'],
    }

    def __init__(self, custom_patterns: Dict = None):
        self.pattern_matcher = PatternMatcher(custom_patterns)
        self.doc_parser = DocumentParser()

    def detect_all(self, file_bytes: bytes) -> DetectionResult:
        """
        Run all detection methods and combine results.

        Args:
            file_bytes: Raw bytes of the Word document

        Returns:
            DetectionResult with all detected insertion points
        """
        result = DetectionResult()
        seen_texts: Set[str] = set()

        # Parse document structure
        structure = self.doc_parser.parse_document(file_bytes)
        full_text = structure.get_full_text()

        # Method 1: Detect explicit placeholders (highest confidence)
        pattern_points = self.detect_pattern_placeholders(structure)
        for point in pattern_points:
            if point.text not in seen_texts:
                result.insertion_points.append(point)
                seen_texts.add(point.text)
        result.detection_stats['pattern_placeholders'] = len(pattern_points)

        # Method 2: Detect highlighted text (from original templer)
        highlight_points = self.detect_highlights(structure)
        for point in highlight_points:
            if point.text not in seen_texts:
                result.insertion_points.append(point)
                seen_texts.add(point.text)
        result.detection_stats['highlights'] = len(highlight_points)

        # Method 3: Detect dynamic values (dates, currencies, percentages)
        dynamic_points = self.detect_dynamic_values(structure, full_text)
        for point in dynamic_points:
            if point.text not in seen_texts:
                result.insertion_points.append(point)
                seen_texts.add(point.text)
        result.detection_stats['dynamic_values'] = len(dynamic_points)

        # Method 4: Detect by context (heading patterns)
        context_points = self.detect_by_context(structure, full_text)
        for point in context_points:
            if point.text not in seen_texts:
                result.insertion_points.append(point)
                seen_texts.add(point.text)
        result.detection_stats['context_based'] = len(context_points)

        # Method 5: Heuristic detection for remaining cases
        heuristic_points = self.detect_by_heuristics(structure, full_text, seen_texts)
        for point in heuristic_points:
            if point.text not in seen_texts:
                result.insertion_points.append(point)
                seen_texts.add(point.text)
        result.detection_stats['heuristics'] = len(heuristic_points)

        # Sort by position and calculate overall confidence
        result.insertion_points.sort(key=lambda p: p.xml_position)
        if result.insertion_points:
            result.overall_confidence = sum(p.confidence for p in result.insertion_points) / len(result.insertion_points)

        return result

    def detect_pattern_placeholders(self, structure: DocumentStructure) -> List[InsertionPoint]:
        """
        Find explicit placeholder patterns like {{x}}, [X], ${x}.
        These have highest confidence.
        """
        points = []
        full_text = structure.get_full_text()

        matches = self.pattern_matcher.find_placeholders(full_text)

        for match in matches:
            context = extract_context_label(full_text, match.start)
            field_type = self.pattern_matcher.infer_field_type(match.value, context)

            points.append(InsertionPoint(
                text=match.full_match,  # Keep the full placeholder
                context=context,
                field_type=field_type,
                detection_method='pattern_placeholder',
                confidence=0.95,
                xml_position=self._find_xml_position(structure.document_xml, match.full_match)
            ))

        return points

    def detect_highlights(self, structure: DocumentStructure) -> List[InsertionPoint]:
        """
        Detect highlighted text in the document (original templer behavior).
        """
        points = []

        for para_idx, para in enumerate(structure.paragraphs):
            for run_idx, run in enumerate(para.runs):
                if run.highlight_color:
                    context = self._get_run_context(para, run_idx)
                    field_type = self.pattern_matcher.infer_field_type(run.text, context)

                    points.append(InsertionPoint(
                        text=run.text,
                        context=context,
                        field_type=field_type,
                        detection_method='highlight',
                        confidence=0.90,
                        xml_position=run.xml_start,
                        paragraph_index=para_idx,
                        run_index=run_idx,
                        is_highlighted=True,
                        highlight_color=run.highlight_color
                    ))

        return points

    def detect_dynamic_values(self, structure: DocumentStructure, full_text: str) -> List[InsertionPoint]:
        """
        Find dynamic values like dates, currencies, percentages.
        These are likely to need replacement even without explicit markers.
        """
        points = []

        # Find dynamic patterns
        matches = self.pattern_matcher.find_dynamic_values(full_text)

        for match in matches:
            context = extract_context_label(full_text, match.start)

            # Score likelihood that this value should be replaced
            likelihood = self.score_replacement_likelihood(match.value, context, match.pattern_type)

            if likelihood >= 0.5:  # Only include if reasonably likely
                points.append(InsertionPoint(
                    text=match.value,
                    context=context,
                    field_type=match.pattern_type,
                    detection_method='dynamic_value',
                    confidence=likelihood,
                    xml_position=self._find_xml_position(structure.document_xml, match.value)
                ))

        return points

    def detect_by_context(self, structure: DocumentStructure, full_text: str) -> List[InsertionPoint]:
        """
        Detect values based on context patterns like "Client Name:", "Amount:".
        """
        points = []

        # Pattern for label: value pairs
        label_value_pattern = r'([A-Z][a-zA-Z\s]+?)\s*:\s*([^\n:]{2,50}?)(?:\n|$|(?=\s{2,}))'

        for match in re.finditer(label_value_pattern, full_text):
            label = match.group(1).strip().lower()
            value = match.group(2).strip()

            if not value or len(value) < 2:
                continue

            # Determine if this label suggests a dynamic field
            field_type = None
            for ftype, keywords in self.CONTEXT_KEYWORDS.items():
                if any(kw in label for kw in keywords):
                    field_type = ftype
                    break

            if field_type:
                # Validate the value matches expected type
                if self._validate_value_type(value, field_type):
                    points.append(InsertionPoint(
                        text=value,
                        context=match.group(1).strip(),
                        field_type=field_type,
                        detection_method='context_label',
                        confidence=0.75,
                        xml_position=self._find_xml_position(structure.document_xml, value)
                    ))

        return points

    def detect_by_heuristics(self, structure: DocumentStructure, full_text: str,
                             already_found: Set[str]) -> List[InsertionPoint]:
        """
        Apply heuristic detection for values that might need replacement.
        Lower confidence but catches edge cases.
        """
        points = []

        # Look for proper names that might be client/advisor names
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for match in re.finditer(name_pattern, full_text):
            name = match.group(1)
            if name in already_found:
                continue

            # Check if it looks like a real name and not a heading
            if self._looks_like_name(name, full_text, match.start()):
                context = extract_context_label(full_text, match.start())
                points.append(InsertionPoint(
                    text=name,
                    context=context,
                    field_type='name',
                    detection_method='heuristic_name',
                    confidence=0.55,
                    xml_position=self._find_xml_position(structure.document_xml, name)
                ))

        # Look for standalone numbers that might be amounts
        amount_pattern = r'\b(\d{1,3}(?:,\d{3})+(?:\.\d{2})?)\b'
        for match in re.finditer(amount_pattern, full_text):
            amount = match.group(1)
            if amount in already_found:
                continue

            context = extract_context_label(full_text, match.start())
            if any(kw in context.lower() for kw in self.CONTEXT_KEYWORDS['currency']):
                points.append(InsertionPoint(
                    text=amount,
                    context=context,
                    field_type='number',
                    detection_method='heuristic_amount',
                    confidence=0.45,
                    xml_position=self._find_xml_position(structure.document_xml, amount)
                ))

        return points

    def score_replacement_likelihood(self, value: str, context: str, pattern_type: str) -> float:
        """
        Score how likely a detected value should be replaced.

        Args:
            value: The detected value
            context: Surrounding context
            pattern_type: Type of pattern that matched

        Returns:
            Score 0-1 indicating likelihood of needing replacement
        """
        score = 0.5  # Base score

        # Pattern type boosts
        type_boosts = {
            'placeholders': 0.45,
            'currency': 0.25,
            'percentage': 0.20,
            'date': 0.20,
            'name_context': 0.30,
            'reference': 0.25,
            'email': 0.30,
            'phone': 0.30,
        }
        score += type_boosts.get(pattern_type, 0)

        # Context boosts
        context_lower = context.lower()
        if any(kw in context_lower for kw in ['client', 'customer', 'name', 'applicant']):
            score += 0.15
        if any(kw in context_lower for kw in ['amount', 'value', 'total', 'balance']):
            score += 0.10
        if any(kw in context_lower for kw in ['date', 'dated', 'expires']):
            score += 0.10

        # Penalty for generic/boilerplate content
        if any(kw in context_lower for kw in ['example', 'sample', 'template', 'footer', 'header']):
            score -= 0.20

        # Penalty for very short or very long values
        if len(value) < 2:
            score -= 0.25
        if len(value) > 100:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _get_run_context(self, paragraph: Paragraph, run_index: int) -> str:
        """Get context from runs before the target run"""
        context_parts = []
        for i in range(max(0, run_index - 3), run_index):
            context_parts.append(paragraph.runs[i].text)
        return ' '.join(context_parts).strip()

    def _find_xml_position(self, xml: str, text: str) -> int:
        """Find approximate position of text in XML"""
        # Escape for regex
        escaped = re.escape(text)
        match = re.search(escaped, xml)
        return match.start() if match else 0

    def _validate_value_type(self, value: str, expected_type: str) -> bool:
        """Validate that a value matches its expected type"""
        if expected_type == 'currency':
            return bool(re.match(r'^[£$€]?\s*[\d,]+(?:\.\d{2})?$', value))
        elif expected_type == 'percentage':
            return bool(re.match(r'^[+-]?\d+\.?\d*\s*%$', value))
        elif expected_type == 'date':
            return bool(re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', value, re.I))
        elif expected_type == 'email':
            return '@' in value and '.' in value
        elif expected_type == 'phone':
            return bool(re.match(r'^[+\d\s\-()]+$', value))
        elif expected_type == 'name':
            return bool(re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', value))
        return True  # Accept by default for other types

    def _looks_like_name(self, name: str, full_text: str, position: int) -> bool:
        """Check if a capitalized phrase looks like a person's name"""
        # Reject if it's a heading or title
        context_before = full_text[max(0, position-20):position].strip()
        if context_before.endswith('\n') or context_before.endswith('.'):
            # Might be a heading
            return False

        # Reject common non-names
        non_names = ['United Kingdom', 'Financial Services', 'Annual Review', 'Client Summary',
                     'Executive Summary', 'Investment Review', 'Risk Profile']
        if name in non_names:
            return False

        # Should be 2-3 words
        words = name.split()
        if len(words) < 2 or len(words) > 4:
            return False

        return True
