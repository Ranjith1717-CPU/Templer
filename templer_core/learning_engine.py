"""
Learning Engine - Learn template structure from blank + filled example pairs
Uses text diffing to automatically discover insertion points.
"""

import re
import difflib
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from .document_parser import DocumentParser, DocumentStructure
from .patterns import PatternMatcher, FIELD_TYPE_PATTERNS


@dataclass
class FieldChange:
    """Represents a detected change between blank and filled templates"""
    blank_text: str              # Original text in blank template
    filled_text: str             # Replacement text in filled example
    context_before: str          # Text before the change
    context_after: str           # Text after the change
    field_type: str              # Inferred field type
    field_name: str              # Suggested field name
    confidence: float            # Confidence in this detection
    position: int                # Position in document


@dataclass
class TemplateSchema:
    """Schema learned from template analysis"""
    fields: List[FieldChange] = field(default_factory=list)
    field_names: Set[str] = field(default_factory=set)
    template_type: str = "unknown"
    total_fields: int = 0
    learning_confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Convert schema to dictionary for JSON serialization"""
        return {
            "fields": [
                {
                    "name": f.field_name,
                    "type": f.field_type,
                    "blank_value": f.blank_text,
                    "example_value": f.filled_text,
                    "context": f.context_before,
                    "confidence": f.confidence
                }
                for f in self.fields
            ],
            "field_names": list(self.field_names),
            "template_type": self.template_type,
            "total_fields": self.total_fields,
            "learning_confidence": self.learning_confidence
        }


class LearningEngine:
    """
    Learn template structure from blank + filled example pairs.
    Uses diff-based detection to find what changed between templates.
    """

    # Minimum length for a change to be considered significant
    MIN_CHANGE_LENGTH = 2

    # Maximum length for a single field (longer = likely paragraph)
    MAX_SINGLE_FIELD_LENGTH = 200

    def __init__(self):
        self.doc_parser = DocumentParser()
        self.pattern_matcher = PatternMatcher()

    def learn_from_example(self, blank_bytes: bytes, filled_bytes: bytes) -> TemplateSchema:
        """
        Learn template schema from a blank template and filled example.

        Args:
            blank_bytes: Raw bytes of the blank template
            filled_bytes: Raw bytes of the filled example

        Returns:
            TemplateSchema describing the discovered fields
        """
        # Parse both documents
        blank_structure = self.doc_parser.parse_document(blank_bytes)
        filled_structure = self.doc_parser.parse_document(filled_bytes)

        # Extract text from both
        blank_text = blank_structure.get_full_text()
        filled_text = filled_structure.get_full_text()

        # Find differences
        changes = self.compute_text_diff(blank_text, filled_text)

        # Map changes back to positions and infer types
        schema = TemplateSchema()
        seen_names: Set[str] = set()

        for change in changes:
            if len(change['blank']) < self.MIN_CHANGE_LENGTH:
                continue

            # Infer field type
            field_type = self.infer_field_type(change['filled'], change['context_before'])

            # Generate field name
            field_name = self.generate_field_name(change['context_before'], field_type, seen_names)
            seen_names.add(field_name)

            field_change = FieldChange(
                blank_text=change['blank'],
                filled_text=change['filled'],
                context_before=change['context_before'],
                context_after=change['context_after'],
                field_type=field_type,
                field_name=field_name,
                confidence=self.calculate_change_confidence(change),
                position=change['position']
            )

            schema.fields.append(field_change)
            schema.field_names.add(field_name)

        schema.total_fields = len(schema.fields)
        schema.template_type = self.infer_template_type(blank_text)

        if schema.fields:
            schema.learning_confidence = sum(f.confidence for f in schema.fields) / len(schema.fields)

        return schema

    def compute_text_diff(self, blank_text: str, filled_text: str) -> List[Dict]:
        """
        Compute differences between blank and filled text.

        Args:
            blank_text: Text from blank template
            filled_text: Text from filled example

        Returns:
            List of change dictionaries
        """
        changes = []

        # Use SequenceMatcher for intelligent diff
        matcher = difflib.SequenceMatcher(None, blank_text, filled_text)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Text was replaced - this is a field
                blank_segment = blank_text[i1:i2]
                filled_segment = filled_text[j1:j2]

                # Get context
                context_before = blank_text[max(0, i1-100):i1].strip()
                context_after = blank_text[i2:min(len(blank_text), i2+50)].strip()

                # Clean up context to last sentence/phrase boundary
                context_before = self._clean_context(context_before)

                changes.append({
                    'blank': blank_segment.strip(),
                    'filled': filled_segment.strip(),
                    'context_before': context_before,
                    'context_after': context_after,
                    'position': i1,
                    'type': 'replace'
                })

            elif tag == 'insert':
                # Text was inserted - might be a field with empty placeholder
                if j2 - j1 > 5:  # Only consider significant insertions
                    filled_segment = filled_text[j1:j2]
                    context_before = blank_text[max(0, i1-100):i1].strip()
                    context_after = blank_text[i1:min(len(blank_text), i1+50)].strip()

                    context_before = self._clean_context(context_before)

                    changes.append({
                        'blank': '',
                        'filled': filled_segment.strip(),
                        'context_before': context_before,
                        'context_after': context_after,
                        'position': i1,
                        'type': 'insert'
                    })

        return changes

    def infer_field_type(self, value: str, context: str) -> str:
        """
        Infer the type of a field from its value and context.

        Args:
            value: The field value from filled example
            context: Surrounding context

        Returns:
            Field type string
        """
        # Check by value patterns
        if re.match(r'^[£$€]', value):
            return 'currency'
        if re.search(r'\d+\.?\d*\s*%', value):
            return 'percentage'
        if re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', value):
            return 'date'
        if re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', value, re.I):
            return 'date'
        if '@' in value and '.' in value:
            return 'email'
        if len(value) > 100:
            return 'paragraph'
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', value):
            return 'name'
        if re.match(r'^[\d,]+(?:\.\d{2})?$', value):
            return 'number'

        # Check context for hints
        context_lower = context.lower()
        if any(kw in context_lower for kw in ['name', 'client', 'advisor', 'dear', 'mr', 'mrs']):
            return 'name'
        if any(kw in context_lower for kw in ['amount', 'value', 'total', 'cost', 'fee', '£']):
            return 'currency'
        if any(kw in context_lower for kw in ['date', 'dated', 'on', 'from']):
            return 'date'
        if any(kw in context_lower for kw in ['percent', 'rate', 'return', '%']):
            return 'percentage'
        if any(kw in context_lower for kw in ['email', 'e-mail']):
            return 'email'
        if any(kw in context_lower for kw in ['phone', 'telephone', 'mobile']):
            return 'phone'
        if any(kw in context_lower for kw in ['address', 'located', 'resident']):
            return 'address'
        if any(kw in context_lower for kw in ['reference', 'ref', 'number', 'account', 'policy']):
            return 'reference'

        return 'text'

    def generate_field_name(self, context: str, field_type: str, existing_names: Set[str]) -> str:
        """
        Generate a meaningful field name from context.

        Args:
            context: Context before the field
            field_type: Inferred field type
            existing_names: Set of already used names

        Returns:
            Unique field name
        """
        # Extract potential label from context
        label_patterns = [
            r'([A-Za-z]+(?:\s+[A-Za-z]+)?)\s*:\s*$',  # "Label: "
            r'([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+$',       # "Label "
        ]

        label = None
        for pattern in label_patterns:
            match = re.search(pattern, context)
            if match:
                label = match.group(1).strip()
                break

        # Generate name from label or type
        if label:
            # Clean and convert to snake_case
            name = re.sub(r'[^a-zA-Z\s]', '', label.lower())
            name = '_'.join(name.split())
        else:
            name = field_type

        # Ensure uniqueness
        base_name = name
        counter = 1
        while name in existing_names:
            name = f"{base_name}_{counter}"
            counter += 1

        return name

    def calculate_change_confidence(self, change: Dict) -> float:
        """
        Calculate confidence score for a detected change.

        Args:
            change: Change dictionary

        Returns:
            Confidence score 0-1
        """
        confidence = 0.7  # Base confidence

        # Boost for clear context
        if ':' in change['context_before']:
            confidence += 0.1

        # Boost for replacement (vs insert)
        if change['type'] == 'replace':
            confidence += 0.1

        # Penalty for very short changes
        if len(change.get('blank', '')) < 3:
            confidence -= 0.1

        # Penalty for very long changes (might be paragraph)
        if len(change.get('filled', '')) > 200:
            confidence -= 0.1

        # Boost if value looks like known type
        if re.match(r'^[£$€]', change.get('filled', '')):
            confidence += 0.1

        return max(0.3, min(1.0, confidence))

    def infer_template_type(self, text: str) -> str:
        """Infer the type of template from its content"""
        text_lower = text.lower()

        if 'annual review' in text_lower or 'yearly review' in text_lower:
            return 'annual_review'
        if 'suitability report' in text_lower or 'suitability' in text_lower:
            return 'suitability_report'
        if 'fact find' in text_lower or 'financial questionnaire' in text_lower:
            return 'fact_find'
        if 'pension transfer' in text_lower:
            return 'pension_transfer'
        if 'investment advice' in text_lower:
            return 'investment_advice'

        return 'general_report'

    def _clean_context(self, context: str) -> str:
        """Clean context string to relevant portion"""
        # Find last sentence or phrase boundary
        for sep in ['. ', ':\n', '\n\n', '•', '-']:
            if sep in context:
                context = context.split(sep)[-1]

        return context.strip()[-100:]  # Limit length

    def build_schema(self, changes: List[Dict]) -> Dict:
        """
        Build a schema dictionary from detected changes.

        Args:
            changes: List of change dictionaries

        Returns:
            Schema dictionary
        """
        schema = {
            'fields': [],
            'field_count': 0,
            'template_type': 'unknown'
        }

        seen_names: Set[str] = set()

        for change in changes:
            field_type = self.infer_field_type(change['filled'], change['context_before'])
            field_name = self.generate_field_name(change['context_before'], field_type, seen_names)
            seen_names.add(field_name)

            schema['fields'].append({
                'name': field_name,
                'type': field_type,
                'blank_value': change['blank'],
                'example_value': change['filled'],
                'context': change['context_before'],
                'position': change['position']
            })

        schema['field_count'] = len(schema['fields'])
        return schema

    def apply_learned_schema(self, template_bytes: bytes, schema: TemplateSchema,
                              new_data: Dict[str, str]) -> bytes:
        """
        Apply a learned schema to a template with new data.

        Args:
            template_bytes: Raw template bytes
            schema: Learned schema
            new_data: Dictionary of field_name -> new_value

        Returns:
            Filled document bytes
        """
        import zipfile
        import io

        # Read the template
        template_io = io.BytesIO(template_bytes)
        output_io = io.BytesIO()

        with zipfile.ZipFile(template_io, 'r') as zin:
            with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.namelist():
                    data = zin.read(item)

                    if item == 'word/document.xml':
                        content = data.decode('utf-8')

                        # Apply each field replacement
                        for field_info in schema.fields:
                            if field_info.field_name in new_data:
                                old_value = field_info.blank_text
                                new_value = new_data[field_info.field_name]

                                # Escape XML
                                new_value_escaped = (str(new_value)
                                    .replace('&', '&amp;')
                                    .replace('<', '&lt;')
                                    .replace('>', '&gt;')
                                    .replace('"', '&quot;')
                                    .replace("'", '&apos;'))

                                content = content.replace(old_value, new_value_escaped)

                        data = content.encode('utf-8')

                    zout.writestr(item, data)

        return output_io.getvalue()
