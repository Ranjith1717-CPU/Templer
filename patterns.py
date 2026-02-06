"""
Regex Patterns for Template Intelligence Engine
Ported from hackathon-word-processor.js with Python enhancements
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PatternMatch:
    """Represents a pattern match with metadata"""
    value: str
    full_match: str
    start: int
    end: int
    pattern_type: str
    confidence: float = 0.8


# =============================================================================
# CORE PATTERN DEFINITIONS
# =============================================================================

PATTERNS = {
    # Template placeholder patterns (explicit markers)
    'placeholders': [
        r'\{\{([^}]+)\}\}',           # {{placeholder}}
        r'\$\{([^}]+)\}',             # ${placeholder}
        r'\[([A-Z_][A-Z_\s]+)\]',     # [PLACEHOLDER]
        r'<<<([^>]+)>>>',             # <<<placeholder>>>
        r'__([A-Z_]+)__',             # __PLACEHOLDER__
        r'\<\<([^>]+)\>\>',           # <<placeholder>>
    ],

    # Currency patterns
    'currency': [
        r'[£$€]\s*[\d,]+(?:\.\d{2})?',                           # £1,234.56
        r'(?:GBP|USD|EUR)\s*[\d,]+(?:\.\d{2})?',                 # GBP 1234.56
        r'[\d,]+(?:\.\d{2})?\s*(?:pounds?|dollars?|euros?)',     # 1234.56 pounds
    ],

    # Percentage patterns
    'percentage': [
        r'[+-]?\d+\.?\d*\s*%',                                   # 5.5%
        r'(?:return|growth|performance|gain|loss)[:\s]*[+-]?\d+\.?\d*\s*%',
    ],

    # Date patterns
    'date': [
        r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',                     # 01/02/2024
        r'\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}',
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}',
    ],

    # Name patterns (context-based)
    'name_context': [
        r'(?:Client|Customer|Name|Dear|Mr\.|Mrs\.|Ms\.|Dr\.)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'Prepared\s+for[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'(?:Applicant|Investor|Account\s+Holder)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ],

    # Email patterns
    'email': [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    ],

    # Phone patterns
    'phone': [
        r'(?:\+44|0)\s*\d{2,4}\s*\d{3,4}\s*\d{3,4}',            # UK phone
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',                        # US format
    ],

    # Address patterns
    'address': [
        r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Road|Street|Avenue|Lane|Drive|Close|Way|Place|Court|Crescent|Gardens)',
    ],

    # Policy/Reference number patterns
    'reference': [
        r'(?:Policy|Account|Reference|Ref|ID)[:\s#]*([A-Z0-9\-]+)',
        r'(?:SIPP|ISA|GIA|Pension)[:\s]*([A-Z0-9\-]+)',
    ],

    # Provider/Company patterns
    'provider': [
        r'(?:Provider|Platform|Wrapper|Company)[:\s]+([A-Za-z\s&]+?)(?:\s*[,\.\n]|$)',
    ],

    # Financial values with context
    'pension_values': [
        r'(?:SIPP|pension|personal\s+pension|occupational\s+pension)[^£€$]*([£$€]\s*[\d,]+(?:\.\d{2})?)',
    ],
    'isa_values': [
        r'(?:ISA|individual\s+savings\s+account)[^£€$]*([£$€]\s*[\d,]+(?:\.\d{2})?)',
    ],
    'portfolio_values': [
        r'(?:portfolio|investment|fund\s+value|current\s+value|total\s+value)[^£€$]*([£$€]\s*[\d,]+(?:\.\d{2})?)',
    ],

    # Risk profile patterns
    'risk_profile': [
        r'(?:risk|attitude|capacity|profile)[^:]*[:\s]*(conservative|moderate|balanced|aggressive|cautious|adventurous|low|medium|high)',
    ],

    # Recommendation patterns
    'recommendations': [
        r'(?:recommend|suggest|advise|propose)[^.!?]*[.!?]',
    ],

    # Meeting dates
    'meeting_dates': [
        r'(?:meeting|appointment|review)[^0-9]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
    ],
}


# Field type inference patterns
FIELD_TYPE_PATTERNS = {
    'currency': r'^[£$€]|^\d+[\d,]*\.\d{2}$|(?:pounds?|dollars?|euros?)$',
    'percentage': r'%$|^[+-]?\d+\.?\d*\s*%',
    'date': r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
    'email': r'@.*\.',
    'phone': r'^\+?\d[\d\s\-()]+$',
    'name': r'^[A-Z][a-z]+\s+[A-Z][a-z]+',
    'paragraph': None,  # Detected by length
}


class PatternMatcher:
    """
    Pattern matching engine for template field detection.
    Provides methods for detecting various types of dynamic content.
    """

    def __init__(self, custom_patterns: Dict = None):
        self.patterns = {**PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # Compile all patterns for performance
        self._compiled = {}
        for pattern_type, patterns in self.patterns.items():
            if isinstance(patterns, list):
                self._compiled[pattern_type] = [
                    re.compile(p, re.IGNORECASE | re.MULTILINE)
                    for p in patterns
                ]
            else:
                self._compiled[pattern_type] = [
                    re.compile(patterns, re.IGNORECASE | re.MULTILINE)
                ]

    def find_all(self, text: str, pattern_types: List[str] = None) -> List[PatternMatch]:
        """
        Find all pattern matches in text.

        Args:
            text: Text to search
            pattern_types: Specific pattern types to search for (None = all)

        Returns:
            List of PatternMatch objects
        """
        matches = []
        types_to_search = pattern_types or list(self._compiled.keys())

        for ptype in types_to_search:
            if ptype not in self._compiled:
                continue

            for pattern in self._compiled[ptype]:
                for match in pattern.finditer(text):
                    # Get captured group or full match
                    value = match.group(1) if match.lastindex else match.group(0)
                    matches.append(PatternMatch(
                        value=value.strip(),
                        full_match=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        pattern_type=ptype,
                        confidence=self._calculate_confidence(ptype, value)
                    ))

        # Sort by position and remove overlapping matches
        matches = self._deduplicate_matches(matches)
        return matches

    def find_placeholders(self, text: str) -> List[PatternMatch]:
        """Find explicit placeholder patterns like {{x}}, [X], ${x}"""
        return self.find_all(text, ['placeholders'])

    def find_dynamic_values(self, text: str) -> List[PatternMatch]:
        """Find dynamic values like dates, currencies, percentages"""
        return self.find_all(text, ['currency', 'percentage', 'date', 'email', 'phone'])

    def find_by_context(self, text: str) -> List[PatternMatch]:
        """Find values based on context (e.g., "Client Name:", "Amount:")"""
        return self.find_all(text, ['name_context', 'reference', 'provider'])

    def infer_field_type(self, value: str, context: str = "") -> str:
        """
        Infer the type of a field from its value and context.

        Args:
            value: The field value
            context: Surrounding text for context

        Returns:
            Field type string (currency, percentage, date, name, text, paragraph)
        """
        # Check by length first
        if len(value) > 100:
            return 'paragraph'

        # Check against patterns
        for field_type, pattern in FIELD_TYPE_PATTERNS.items():
            if pattern and re.search(pattern, value, re.IGNORECASE):
                return field_type

        # Check context for hints
        context_lower = context.lower()
        if any(kw in context_lower for kw in ['amount', 'value', 'cost', 'fee', 'price']):
            return 'currency'
        if any(kw in context_lower for kw in ['date', 'when', 'time']):
            return 'date'
        if any(kw in context_lower for kw in ['name', 'client', 'advisor']):
            return 'name'
        if any(kw in context_lower for kw in ['percent', 'rate', 'return']):
            return 'percentage'

        return 'text'

    def _calculate_confidence(self, pattern_type: str, value: str) -> float:
        """Calculate confidence score for a match"""
        base_confidence = {
            'placeholders': 0.95,      # Explicit placeholders are very confident
            'currency': 0.85,
            'percentage': 0.85,
            'date': 0.80,
            'email': 0.90,
            'phone': 0.75,
            'name_context': 0.70,
            'reference': 0.75,
            'provider': 0.65,
            'pension_values': 0.80,
            'isa_values': 0.80,
            'portfolio_values': 0.80,
            'risk_profile': 0.70,
            'recommendations': 0.60,
            'meeting_dates': 0.75,
            'address': 0.70,
        }.get(pattern_type, 0.5)

        # Adjust based on value characteristics
        if len(value) < 2:
            base_confidence -= 0.2
        if len(value) > 200:
            base_confidence -= 0.1

        return max(0.1, min(1.0, base_confidence))

    def _deduplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Remove overlapping matches, keeping highest confidence"""
        if not matches:
            return []

        # Sort by position
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))

        result = []
        last_end = -1

        for match in sorted_matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end
            elif match.confidence > result[-1].confidence:
                # Replace with higher confidence match
                result[-1] = match
                last_end = match.end

        return result


def extract_context_label(text: str, position: int, max_chars: int = 100) -> str:
    """
    Extract the context/label before a value position.
    Useful for understanding what a placeholder represents.

    Args:
        text: Full text
        position: Position of the value
        max_chars: Maximum characters to look back

    Returns:
        Context string (e.g., "Client Name:", "Portfolio Value:")
    """
    start = max(0, position - max_chars)
    context = text[start:position]

    # Find the last label pattern
    label_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*:\s*$',  # "Client Name: "
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+$',       # "Client Name "
    ]

    for pattern in label_patterns:
        match = re.search(pattern, context)
        if match:
            return match.group(1).strip()

    # Return last line or phrase
    lines = context.strip().split('\n')
    return lines[-1].strip()[-50:] if lines else ""
