"""
Conditional Handler for Template Processing
Supports IF/THEN logic for showing/hiding sections based on data.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ConditionalBlock:
    """Represents a conditional section in a template"""
    expression: str              # The condition expression
    start_marker: str            # Full start marker text
    end_marker: str              # Full end marker text
    content: str                 # Content between markers
    start_pos: int               # Start position in document
    end_pos: int                 # End position in document
    block_type: str = 'if'       # if, unless, for


class ConditionalHandler:
    """
    Handle conditional sections in templates.

    Supported formats:
    - [[IF condition]]...[[/IF]]
    - [[IF value > 100000]]...[[/IF]]
    - {% if condition %}...{% endif %}
    - {{#if condition}}...{{/if}}
    - [[UNLESS condition]]...[[/UNLESS]]
    - [[FOR item in items]]...[[/FOR]]
    """

    # Pattern definitions for different conditional syntaxes
    CONDITIONAL_PATTERNS = [
        # [[IF condition]]...[[/IF]]
        {
            'start': r'\[\[IF\s+([^\]]+)\]\]',
            'end': r'\[\[/IF\]\]',
            'type': 'if'
        },
        # [[UNLESS condition]]...[[/UNLESS]]
        {
            'start': r'\[\[UNLESS\s+([^\]]+)\]\]',
            'end': r'\[\[/UNLESS\]\]',
            'type': 'unless'
        },
        # {% if condition %}...{% endif %}
        {
            'start': r'\{%\s*if\s+([^%]+)\s*%\}',
            'end': r'\{%\s*endif\s*%\}',
            'type': 'if'
        },
        # {{#if condition}}...{{/if}}
        {
            'start': r'\{\{#if\s+([^}]+)\}\}',
            'end': r'\{\{/if\}\}',
            'type': 'if'
        },
        # [[FOR item in items]]...[[/FOR]]
        {
            'start': r'\[\[FOR\s+([^\]]+)\]\]',
            'end': r'\[\[/FOR\]\]',
            'type': 'for'
        },
    ]

    # Comparison operators supported in conditions
    OPERATORS = {
        '>=': lambda a, b: float(a) >= float(b),
        '<=': lambda a, b: float(a) <= float(b),
        '>': lambda a, b: float(a) > float(b),
        '<': lambda a, b: float(a) < float(b),
        '==': lambda a, b: str(a).lower() == str(b).lower(),
        '!=': lambda a, b: str(a).lower() != str(b).lower(),
        '=': lambda a, b: str(a).lower() == str(b).lower(),
    }

    def __init__(self):
        pass

    def apply_conditionals(self, document_xml: str, data: Dict[str, Any]) -> str:
        """
        Apply all conditional logic to a document.

        Args:
            document_xml: Document XML with conditional markers
            data: Data dictionary for evaluating conditions

        Returns:
            Modified document XML with conditionals processed
        """
        # Find all conditional blocks
        conditionals = self.detect_conditional_markers(document_xml)

        # Sort by position (reverse order to avoid offset issues)
        conditionals.sort(key=lambda c: c.start_pos, reverse=True)

        # Process each conditional
        for cond in conditionals:
            if cond.block_type == 'for':
                # Handle FOR loops
                document_xml = self._process_for_loop(document_xml, cond, data)
            else:
                # Handle IF/UNLESS
                result = self.evaluate_condition(cond.expression, data)

                # For UNLESS, invert the result
                if cond.block_type == 'unless':
                    result = not result

                if not result:
                    # Remove section
                    document_xml = self.remove_section(document_xml, cond.start_pos, cond.end_pos)
                else:
                    # Keep content but remove markers
                    document_xml = self.remove_markers_only(document_xml, cond)

        return document_xml

    def detect_conditional_markers(self, document_xml: str) -> List[ConditionalBlock]:
        """
        Find all conditional markers in the document.

        Args:
            document_xml: Document XML to search

        Returns:
            List of ConditionalBlock objects
        """
        blocks = []

        for pattern_def in self.CONDITIONAL_PATTERNS:
            start_pattern = pattern_def['start']
            end_pattern = pattern_def['end']
            block_type = pattern_def['type']

            # Find all start markers
            for start_match in re.finditer(start_pattern, document_xml, re.IGNORECASE):
                expression = start_match.group(1).strip()
                start_marker = start_match.group(0)
                start_pos = start_match.start()

                # Find matching end marker
                search_start = start_match.end()
                end_match = re.search(end_pattern, document_xml[search_start:], re.IGNORECASE)

                if end_match:
                    end_marker = end_match.group(0)
                    end_pos = search_start + end_match.end()
                    content = document_xml[start_match.end():search_start + end_match.start()]

                    blocks.append(ConditionalBlock(
                        expression=expression,
                        start_marker=start_marker,
                        end_marker=end_marker,
                        content=content,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        block_type=block_type
                    ))

        return blocks

    def evaluate_condition(self, expression: str, data: Dict[str, Any]) -> bool:
        """
        Evaluate a condition expression against data.

        Args:
            expression: Condition expression (e.g., "has_pension", "value > 100000")
            data: Data dictionary

        Returns:
            Boolean result of condition
        """
        expression = expression.strip()

        # Check for comparison operators
        for op, func in self.OPERATORS.items():
            if op in expression:
                parts = expression.split(op, 1)
                if len(parts) == 2:
                    left = self._get_value(parts[0].strip(), data)
                    right = self._get_value(parts[1].strip(), data)

                    try:
                        return func(left, right)
                    except (ValueError, TypeError):
                        return False

        # Check for boolean operators
        if ' and ' in expression.lower():
            parts = re.split(r'\s+and\s+', expression, flags=re.IGNORECASE)
            return all(self.evaluate_condition(p, data) for p in parts)

        if ' or ' in expression.lower():
            parts = re.split(r'\s+or\s+', expression, flags=re.IGNORECASE)
            return any(self.evaluate_condition(p, data) for p in parts)

        # Check for NOT operator
        if expression.lower().startswith('not '):
            inner_expr = expression[4:].strip()
            return not self.evaluate_condition(inner_expr, data)

        # Simple truthiness check
        value = self._get_value(expression, data)
        return self._is_truthy(value)

    def remove_section(self, document_xml: str, start_pos: int, end_pos: int) -> str:
        """
        Remove a section from the document.

        Args:
            document_xml: Document XML
            start_pos: Start position to remove
            end_pos: End position to remove

        Returns:
            Modified document XML
        """
        return document_xml[:start_pos] + document_xml[end_pos:]

    def remove_markers_only(self, document_xml: str, block: ConditionalBlock) -> str:
        """
        Remove only the conditional markers, keeping the content.

        Args:
            document_xml: Document XML
            block: ConditionalBlock to process

        Returns:
            Modified document XML
        """
        # Remove start marker
        result = document_xml.replace(block.start_marker, '', 1)

        # Remove end marker
        result = result.replace(block.end_marker, '', 1)

        return result

    def _process_for_loop(self, document_xml: str, block: ConditionalBlock,
                          data: Dict[str, Any]) -> str:
        """
        Process a FOR loop block.

        Args:
            document_xml: Document XML
            block: FOR loop block
            data: Data dictionary

        Returns:
            Modified document XML
        """
        # Parse FOR expression: "item in items"
        match = re.match(r'(\w+)\s+in\s+(\w+)', block.expression)
        if not match:
            return self.remove_markers_only(document_xml, block)

        item_var = match.group(1)
        collection_var = match.group(2)

        # Get collection from data
        collection = data.get(collection_var, [])
        if not isinstance(collection, (list, tuple)):
            collection = []

        # Generate repeated content
        repeated_content = []
        for item in collection:
            # Create context with loop variable
            loop_data = {**data, item_var: item}

            # Replace variables in content
            content = block.content
            content = self._replace_variables(content, loop_data)
            repeated_content.append(content)

        # Replace the entire block with repeated content
        full_block = document_xml[block.start_pos:block.end_pos]
        replacement = ''.join(repeated_content)

        return document_xml.replace(full_block, replacement, 1)

    def _get_value(self, expression: str, data: Dict[str, Any]) -> Any:
        """
        Get a value from data, supporting nested paths.

        Args:
            expression: Variable expression (e.g., "client.name", "value")
            data: Data dictionary

        Returns:
            Value or expression itself if not a variable
        """
        # Remove quotes if it's a string literal
        if (expression.startswith('"') and expression.endswith('"')) or \
           (expression.startswith("'") and expression.endswith("'")):
            return expression[1:-1]

        # Try as number
        try:
            if '.' in expression:
                return float(expression)
            return int(expression)
        except ValueError:
            pass

        # Try as variable path
        parts = expression.split('.')
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value

    def _is_truthy(self, value: Any) -> bool:
        """Check if a value is truthy"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.lower() not in ('', 'false', 'no', '0', 'none', 'null')
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return True

    def _replace_variables(self, content: str, data: Dict[str, Any]) -> str:
        """
        Replace variable placeholders in content.

        Args:
            content: Content with placeholders
            data: Data dictionary

        Returns:
            Content with replaced values
        """
        # Replace {{variable}} patterns
        def replacer(match):
            var_name = match.group(1).strip()
            value = self._get_value(var_name, data)
            return str(value) if value is not None else match.group(0)

        content = re.sub(r'\{\{([^}]+)\}\}', replacer, content)

        return content


def create_conditional_handler():
    """Factory function for ConditionalHandler"""
    return ConditionalHandler()
