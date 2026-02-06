"""
Validation Engine for Template Processing
Provides confidence scoring and output validation.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class FieldValidation:
    """Validation result for a single field"""
    field_name: str
    confidence: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class DocumentValidation:
    """Complete document validation result"""
    is_valid: bool = True
    overall_confidence: float = 0.0
    field_validations: List[FieldValidation] = field(default_factory=list)
    structural_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "overall_confidence": self.overall_confidence,
            "fields": [
                {
                    "name": fv.field_name,
                    "confidence": fv.confidence,
                    "is_valid": fv.is_valid,
                    "issues": fv.issues,
                    "warnings": fv.warnings
                }
                for fv in self.field_validations
            ],
            "structural_issues": self.structural_issues,
            "warnings": self.warnings
        }


class ValidationEngine:
    """
    Validate template processing results and score confidence.
    """

    # Expected field types and their validation patterns
    FIELD_VALIDATORS = {
        'currency': {
            'pattern': r'^[£$€]?\s*[\d,]+(?:\.\d{2})?$',
            'error': 'Invalid currency format'
        },
        'percentage': {
            'pattern': r'^[+-]?\d+\.?\d*\s*%$',
            'error': 'Invalid percentage format'
        },
        'date': {
            'pattern': r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
            'error': 'Invalid date format'
        },
        'email': {
            'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'error': 'Invalid email format'
        },
        'phone': {
            'pattern': r'^[+\d\s\-()]+$',
            'error': 'Invalid phone format'
        },
        'name': {
            'pattern': r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$',
            'error': 'Invalid name format'
        },
    }

    def __init__(self):
        pass

    def validate_mappings(self, mappings: Dict[str, str],
                          field_types: Dict[str, str] = None) -> DocumentValidation:
        """
        Validate field mappings and score confidence.

        Args:
            mappings: Dictionary of original -> replacement values
            field_types: Optional dictionary of field -> expected type

        Returns:
            DocumentValidation result
        """
        result = DocumentValidation()
        field_types = field_types or {}

        for original, replacement in mappings.items():
            field_name = self._extract_field_name(original)
            field_type = field_types.get(field_name, 'text')

            validation = self._validate_field(field_name, replacement, field_type)
            result.field_validations.append(validation)

            if not validation.is_valid:
                result.is_valid = False

        # Calculate overall confidence
        if result.field_validations:
            result.overall_confidence = sum(
                fv.confidence for fv in result.field_validations
            ) / len(result.field_validations)
        else:
            result.overall_confidence = 1.0

        return result

    def validate_document_structure(self, document_xml: str) -> DocumentValidation:
        """
        Validate the structural integrity of output document.

        Args:
            document_xml: Generated document XML

        Returns:
            DocumentValidation with structural issues
        """
        result = DocumentValidation()

        # Check for unclosed tags
        unclosed = self._check_unclosed_tags(document_xml)
        if unclosed:
            result.structural_issues.extend(unclosed)
            result.is_valid = False

        # Check for invalid XML characters
        invalid_chars = self._check_invalid_characters(document_xml)
        if invalid_chars:
            result.structural_issues.extend(invalid_chars)
            result.is_valid = False

        # Check for orphaned markers
        orphaned = self._check_orphaned_markers(document_xml)
        if orphaned:
            result.warnings.extend(orphaned)

        # Check document can be parsed
        if not self._check_parseable(document_xml):
            result.structural_issues.append("Document XML is not well-formed")
            result.is_valid = False

        result.overall_confidence = 1.0 if result.is_valid else 0.5

        return result

    def validate_against_example(self, filled_xml: str, example_xml: str) -> float:
        """
        Compare filled document against a known good example.

        Args:
            filled_xml: Generated document XML
            example_xml: Known good example XML

        Returns:
            Similarity score 0-1
        """
        # Extract text from both
        filled_text = self._extract_text(filled_xml)
        example_text = self._extract_text(example_xml)

        # Calculate similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, filled_text, example_text).ratio()

        return similarity

    def score_extraction(self, mappings: Dict[str, str],
                         detection_method: str = 'auto') -> Dict[str, float]:
        """
        Score confidence for each extraction.

        Args:
            mappings: Dictionary of field -> value
            detection_method: How fields were detected

        Returns:
            Dictionary of field -> confidence score
        """
        scores = {}
        method_base_scores = {
            'pattern_placeholder': 0.95,
            'highlight': 0.90,
            'context_label': 0.75,
            'dynamic_value': 0.70,
            'heuristic': 0.50,
            'learning': 0.80,
            'auto': 0.70
        }

        base_score = method_base_scores.get(detection_method, 0.5)

        for field_name, value in mappings.items():
            score = base_score

            # Adjust based on value characteristics
            if value and len(value) >= 2:
                score += 0.05

            if self._looks_like_known_pattern(value):
                score += 0.10

            if self._is_placeholder_like(value):
                score -= 0.20  # Value wasn't replaced

            if len(value) > 200:
                score -= 0.05  # Suspiciously long

            scores[field_name] = max(0.1, min(1.0, score))

        return scores

    def _validate_field(self, field_name: str, value: str,
                       expected_type: str) -> FieldValidation:
        """Validate a single field"""
        validation = FieldValidation(field_name=field_name, confidence=0.8)

        if not value or not value.strip():
            validation.issues.append("Field is empty")
            validation.is_valid = False
            validation.confidence = 0.2
            return validation

        # Check against expected type
        if expected_type in self.FIELD_VALIDATORS:
            validator = self.FIELD_VALIDATORS[expected_type]
            if not re.match(validator['pattern'], value, re.IGNORECASE):
                validation.warnings.append(validator['error'])
                validation.confidence -= 0.2

        # Check for placeholder remnants
        if self._is_placeholder_like(value):
            validation.issues.append("Value appears to be an unfilled placeholder")
            validation.is_valid = False
            validation.confidence = 0.1

        # Check for unrealistic values
        if expected_type == 'currency':
            amount = self._parse_currency(value)
            if amount is not None and amount > 100000000:
                validation.warnings.append("Unusually large amount")
                validation.confidence -= 0.1

        return validation

    def _extract_field_name(self, original_text: str) -> str:
        """Extract a field name from original placeholder text"""
        # Remove placeholder markers
        cleaned = re.sub(r'[\[\]\{\}\$\<\>]', '', original_text)
        # Convert to snake_case
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
        return '_'.join(cleaned.lower().split())[:50]

    def _check_unclosed_tags(self, xml: str) -> List[str]:
        """Check for unclosed XML tags"""
        issues = []

        # Common Word XML elements
        elements = ['w:p', 'w:r', 'w:t', 'w:tbl', 'w:tr', 'w:tc']

        for element in elements:
            opens = len(re.findall(f'<{element}[^>]*>', xml))
            closes = len(re.findall(f'</{element}>', xml))

            if opens != closes:
                issues.append(f"Mismatched {element} tags: {opens} opens, {closes} closes")

        return issues

    def _check_invalid_characters(self, xml: str) -> List[str]:
        """Check for invalid XML characters"""
        issues = []

        # Check for unescaped special characters in text content
        text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'
        for match in re.finditer(text_pattern, xml):
            text = match.group(1)
            if '&' in text and not re.search(r'&(amp|lt|gt|quot|apos);', text):
                issues.append(f"Unescaped ampersand in: {text[:50]}")
            if '<' in text or '>' in text:
                issues.append(f"Unescaped angle bracket in: {text[:50]}")

        return issues[:5]  # Limit to first 5 issues

    def _check_orphaned_markers(self, xml: str) -> List[str]:
        """Check for unfilled placeholder markers"""
        warnings = []

        patterns = [
            r'\{\{[^}]+\}\}',
            r'\$\{[^}]+\}',
            r'\[\[[A-Z_]+\]\]',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, xml)
            if matches:
                warnings.append(f"Unfilled placeholders found: {matches[:3]}")

        return warnings

    def _check_parseable(self, xml: str) -> bool:
        """Check if XML can be parsed"""
        try:
            # Simple check - look for document structure
            has_body = '<w:body>' in xml and '</w:body>' in xml
            has_doc = '<w:document' in xml
            return has_body or has_doc or '<w:p' in xml
        except Exception:
            return False

    def _extract_text(self, xml: str) -> str:
        """Extract plain text from XML"""
        text_pattern = r'<w:t[^>]*>([^<]*)</w:t>'
        matches = re.findall(text_pattern, xml)
        return ' '.join(matches)

    def _looks_like_known_pattern(self, value: str) -> bool:
        """Check if value matches a known pattern (date, currency, etc.)"""
        patterns = [
            r'^[£$€]\s*[\d,]+',      # Currency
            r'\d+\s*%',               # Percentage
            r'\d{1,2}[/\-]\d{1,2}',   # Date
            r'@.*\.',                  # Email
        ]

        return any(re.search(p, value) for p in patterns)

    def _is_placeholder_like(self, value: str) -> bool:
        """Check if value looks like an unfilled placeholder"""
        placeholder_patterns = [
            r'^\{\{.*\}\}$',
            r'^\$\{.*\}$',
            r'^\[.*\]$',
            r'^<<.*>>$',
            r'^__.*__$',
            r'^\[PLACEHOLDER',
            r'^N/A$',
            r'^TBD$',
            r'^INSERT',
            r'^ENTER',
        ]

        return any(re.match(p, value, re.IGNORECASE) for p in placeholder_patterns)

    def _parse_currency(self, value: str) -> Optional[float]:
        """Parse currency value to float"""
        try:
            cleaned = re.sub(r'[£$€,\s]', '', value)
            return float(cleaned)
        except (ValueError, TypeError):
            return None


def create_validation_engine():
    """Factory function for ValidationEngine"""
    return ValidationEngine()
