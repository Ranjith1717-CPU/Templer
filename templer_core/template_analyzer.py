"""
Template Analyzer - Automatically identify static vs dynamic content in templates.

This is the core solution to the template setup problem:
- Takes a RAW template (firm's approved Word doc with no placeholders)
- Analyzes content to identify what's static vs dynamic
- Outputs a NEW template with {{placeholders}} inserted
- Generates prompt hints for LLM-generated sections

Reduces 4-hour manual setup to minutes of review.
"""

import re
import io
import zipfile
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """Classification of content sections"""
    STATIC = "static"           # Never changes (legal, headers, boilerplate)
    DYNAMIC_VALUE = "dynamic_value"     # Simple replacement (name, date, amount)
    DYNAMIC_LLM = "dynamic_llm"         # LLM-generated content (recommendations, assessments)
    TABLE_HEADER = "table_header"       # Table headers (usually static)
    TABLE_DATA = "table_data"           # Table data rows (usually dynamic)


@dataclass
class AnalyzedSection:
    """Represents an analyzed section of the document"""
    text: str
    content_type: ContentType
    confidence: float
    placeholder_name: Optional[str] = None
    prompt_hint: Optional[str] = None
    reasoning: str = ""
    position: int = 0
    heading_context: Optional[str] = None

    # For dynamic values
    value_type: Optional[str] = None  # name, date, currency, percentage, etc.


@dataclass
class AnalysisResult:
    """Complete analysis result for a template"""
    sections: List[AnalyzedSection] = field(default_factory=list)
    static_count: int = 0
    dynamic_value_count: int = 0
    dynamic_llm_count: int = 0
    overall_confidence: float = 0.0
    template_type: str = "unknown"

    def get_dynamic_sections(self) -> List[AnalyzedSection]:
        return [s for s in self.sections if s.content_type != ContentType.STATIC]

    def get_placeholders(self) -> Dict[str, str]:
        """Get mapping of placeholder names to prompt hints"""
        return {
            s.placeholder_name: s.prompt_hint or s.text[:50]
            for s in self.sections
            if s.placeholder_name
        }


class TemplateAnalyzer:
    """
    Analyzes raw templates to identify static vs dynamic content.

    The core problem this solves:
    - Firm gives template with NO markers
    - We need to figure out what stays the same vs what changes per client
    - Output a template with proper placeholders
    """

    # Keywords indicating STATIC content (legal, compliance, boilerplate)
    STATIC_INDICATORS = {
        'legal': [
            'regulated by', 'fca', 'financial conduct authority',
            'authorised', 'authorized', 'disclaimer', 'terms and conditions',
            'privacy', 'data protection', 'gdpr', 'confidential',
            'copyright', 'all rights reserved', 'registered office',
            'company number', 'vat number', 'registered in england',
        ],
        'boilerplate': [
            'important information', 'risk warning', 'past performance',
            'capital at risk', 'tax treatment', 'legislation',
            'this document', 'for professional', 'not intended',
            'seek advice', 'independent advice',
        ],
        'headers': [
            'contents', 'table of contents', 'appendix', 'appendices',
            'page', 'document reference', 'version',
        ],
    }

    # Keywords indicating DYNAMIC content that needs LLM generation
    DYNAMIC_LLM_INDICATORS = {
        'recommendations': [
            'recommendation', 'we recommend', 'our advice', 'suggest',
            'proposed', 'action', 'next steps', 'going forward',
        ],
        'assessments': [
            'assessment', 'analysis', 'evaluation', 'review of',
            'risk profile', 'attitude to risk', 'capacity for loss',
            'suitability', 'appropriateness',
        ],
        'rationale': [
            'rationale', 'reasoning', 'because', 'therefore',
            'as a result', 'given your', 'based on your',
            'considering your', 'taking into account',
        ],
        'circumstances': [
            'your circumstances', 'your situation', 'personal details',
            'your objectives', 'your goals', 'your needs',
            'financial position', 'current position',
        ],
        'summary': [
            'executive summary', 'summary', 'overview', 'key points',
            'highlights', 'in summary',
        ],
    }

    # Headings that typically contain dynamic LLM content
    DYNAMIC_HEADINGS = [
        'recommendation', 'advice', 'assessment', 'analysis',
        'suitability', 'rationale', 'summary', 'overview',
        'circumstances', 'objectives', 'goals', 'position',
        'review', 'commentary', 'discussion', 'conclusion',
    ]

    # Headings that typically contain static content
    STATIC_HEADINGS = [
        'disclaimer', 'important information', 'risk warning',
        'terms', 'conditions', 'legal', 'regulatory',
        'about us', 'contact', 'appendix', 'glossary',
        'definitions', 'notes', 'references',
    ]

    def __init__(self):
        self.current_heading = None

    def analyze_template(self, file_bytes: bytes) -> AnalysisResult:
        """
        Analyze a raw template to identify static vs dynamic content.

        Args:
            file_bytes: Raw bytes of the Word document

        Returns:
            AnalysisResult with classified sections
        """
        result = AnalysisResult()

        # Parse document
        document_xml, full_text = self._parse_document(file_bytes)

        # Detect template type
        result.template_type = self._detect_template_type(full_text)

        # Extract and analyze paragraphs
        paragraphs = self._extract_paragraphs(document_xml)

        position = 0
        self.current_heading = None

        for para_xml, para_text in paragraphs:
            if not para_text.strip():
                continue

            # Check if this is a heading
            is_heading = self._is_heading(para_xml, para_text)
            if is_heading:
                self.current_heading = para_text.strip()

            # Analyze the paragraph
            section = self._analyze_paragraph(para_text, para_xml, position)
            section.heading_context = self.current_heading

            result.sections.append(section)
            position += 1

            # Update counts
            if section.content_type == ContentType.STATIC:
                result.static_count += 1
            elif section.content_type == ContentType.DYNAMIC_VALUE:
                result.dynamic_value_count += 1
            elif section.content_type == ContentType.DYNAMIC_LLM:
                result.dynamic_llm_count += 1

        # Calculate overall confidence
        if result.sections:
            result.overall_confidence = sum(s.confidence for s in result.sections) / len(result.sections)

        return result

    def generate_template(self, file_bytes: bytes, analysis: AnalysisResult = None) -> bytes:
        """
        Generate a new template with placeholders inserted.

        Args:
            file_bytes: Original document bytes
            analysis: Optional pre-computed analysis

        Returns:
            New document bytes with placeholders
        """
        if analysis is None:
            analysis = self.analyze_template(file_bytes)

        # Read original document
        template_io = io.BytesIO(file_bytes)
        output_io = io.BytesIO()

        with zipfile.ZipFile(template_io, 'r') as zin:
            with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.namelist():
                    data = zin.read(item)

                    if item == 'word/document.xml':
                        content = data.decode('utf-8')

                        # Replace dynamic sections with placeholders
                        for section in analysis.sections:
                            if section.content_type != ContentType.STATIC and section.placeholder_name:
                                # Create placeholder text
                                if section.content_type == ContentType.DYNAMIC_LLM:
                                    placeholder = f"{{{{LLM:{section.placeholder_name}}}}}"
                                else:
                                    placeholder = f"{{{{{section.placeholder_name}}}}}"

                                # Replace in content (careful with XML)
                                content = self._safe_replace(content, section.text, placeholder)

                        data = content.encode('utf-8')

                    zout.writestr(item, data)

        return output_io.getvalue()

    def generate_prompt_config(self, analysis: AnalysisResult) -> Dict:
        """
        Generate configuration for LLM prompts based on analysis.

        Returns a config that can be used to drive the LLM generation.
        """
        config = {
            "template_type": analysis.template_type,
            "placeholders": {},
            "llm_sections": [],
            "value_sections": [],
        }

        for section in analysis.sections:
            if section.content_type == ContentType.DYNAMIC_LLM:
                config["llm_sections"].append({
                    "name": section.placeholder_name,
                    "prompt_hint": section.prompt_hint,
                    "heading_context": section.heading_context,
                    "original_text_sample": section.text[:200],
                })
                config["placeholders"][section.placeholder_name] = {
                    "type": "llm",
                    "prompt": section.prompt_hint,
                }
            elif section.content_type == ContentType.DYNAMIC_VALUE:
                config["value_sections"].append({
                    "name": section.placeholder_name,
                    "value_type": section.value_type,
                    "context": section.heading_context,
                })
                config["placeholders"][section.placeholder_name] = {
                    "type": section.value_type,
                    "prompt": f"Extract {section.value_type} from input data",
                }

        return config

    def _parse_document(self, file_bytes: bytes) -> Tuple[str, str]:
        """Parse Word document and return XML and full text"""
        document_xml = ""
        full_text = ""

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                if 'word/document.xml' in zf.namelist():
                    document_xml = zf.read('word/document.xml').decode('utf-8')
                    text_matches = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', document_xml)
                    full_text = ' '.join(text_matches)
        except Exception as e:
            raise ValueError(f"Failed to parse document: {e}")

        return document_xml, full_text

    def _extract_paragraphs(self, document_xml: str) -> List[Tuple[str, str]]:
        """Extract paragraphs with their XML and text"""
        paragraphs = []

        para_pattern = r'<w:p[^>]*>(.*?)</w:p>'
        for match in re.finditer(para_pattern, document_xml, re.DOTALL):
            para_xml = match.group(0)
            text_matches = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', para_xml)
            para_text = ''.join(text_matches)

            if para_text.strip():
                paragraphs.append((para_xml, para_text))

        return paragraphs

    def _is_heading(self, para_xml: str, para_text: str) -> bool:
        """Check if a paragraph is a heading"""
        # Check for heading style
        if re.search(r'<w:pStyle w:val="Heading\d?"', para_xml):
            return True

        # Check for bold short text (likely heading)
        if '<w:b/>' in para_xml or '<w:b ' in para_xml:
            if len(para_text.strip()) < 100:
                return True

        # Check for all caps short text
        if para_text.isupper() and len(para_text.strip()) < 50:
            return True

        return False

    def _analyze_paragraph(self, text: str, para_xml: str, position: int) -> AnalyzedSection:
        """Analyze a single paragraph to classify it"""
        text_lower = text.lower().strip()

        # Check if under a known heading type
        heading_influence = self._get_heading_influence()

        # Score for static content
        static_score = self._score_static(text_lower)

        # Score for dynamic LLM content
        llm_score = self._score_dynamic_llm(text_lower)

        # Score for dynamic values
        value_score, value_type = self._score_dynamic_value(text)

        # Apply heading influence
        if heading_influence == 'static':
            static_score += 0.3
        elif heading_influence == 'dynamic':
            llm_score += 0.3

        # Determine classification
        if static_score > max(llm_score, value_score) and static_score > 0.3:
            return AnalyzedSection(
                text=text,
                content_type=ContentType.STATIC,
                confidence=min(0.95, static_score),
                reasoning="Identified as static content (legal/boilerplate/header)",
                position=position,
            )
        elif llm_score > value_score and llm_score > 0.3:
            placeholder_name = self._generate_placeholder_name(text, 'llm')
            prompt_hint = self._generate_prompt_hint(text)
            return AnalyzedSection(
                text=text,
                content_type=ContentType.DYNAMIC_LLM,
                confidence=min(0.9, llm_score),
                placeholder_name=placeholder_name,
                prompt_hint=prompt_hint,
                reasoning="Identified as LLM-generated content (recommendations/assessments)",
                position=position,
            )
        elif value_score > 0.3:
            placeholder_name = self._generate_placeholder_name(text, value_type)
            return AnalyzedSection(
                text=text,
                content_type=ContentType.DYNAMIC_VALUE,
                confidence=min(0.9, value_score),
                placeholder_name=placeholder_name,
                value_type=value_type,
                reasoning=f"Identified as dynamic value ({value_type})",
                position=position,
            )
        else:
            # Default to static if unclear
            return AnalyzedSection(
                text=text,
                content_type=ContentType.STATIC,
                confidence=0.5,
                reasoning="Defaulted to static (no strong indicators)",
                position=position,
            )

    def _score_static(self, text_lower: str) -> float:
        """Score how likely this is static content"""
        score = 0.0

        for category, keywords in self.STATIC_INDICATORS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.2
                    if score >= 0.8:
                        return score

        # Long paragraphs with no specific client references tend to be static
        if len(text_lower) > 200:
            has_client_ref = any(w in text_lower for w in ['you', 'your', 'client'])
            if not has_client_ref:
                score += 0.2

        return min(1.0, score)

    def _score_dynamic_llm(self, text_lower: str) -> float:
        """Score how likely this needs LLM generation"""
        score = 0.0

        for category, keywords in self.DYNAMIC_LLM_INDICATORS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.15
                    if score >= 0.8:
                        return score

        # Check for client-specific language
        client_refs = ['you', 'your', 'client']
        client_ref_count = sum(1 for ref in client_refs if ref in text_lower)
        score += client_ref_count * 0.1

        # Medium-length paragraphs with recommendations/assessments
        if 50 < len(text_lower) < 500:
            score += 0.1

        return min(1.0, score)

    def _score_dynamic_value(self, text: str) -> Tuple[float, Optional[str]]:
        """Score how likely this contains dynamic values and identify type"""
        # Check for currency
        if re.search(r'[£$€]\s*[\d,]+(?:\.\d{2})?', text):
            return 0.8, 'currency'

        # Check for percentage
        if re.search(r'\d+\.?\d*\s*%', text):
            return 0.7, 'percentage'

        # Check for date
        if re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text):
            return 0.8, 'date'
        if re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}', text, re.I):
            return 0.8, 'date'

        # Check for name patterns with context
        if re.search(r'(?:Client|Name|Dear|Mr\.|Mrs\.|Ms\.)[:\s]+[A-Z][a-z]+\s+[A-Z][a-z]+', text):
            return 0.85, 'name'

        # Check for reference numbers
        if re.search(r'(?:Reference|Ref|Policy|Account)[:\s#]*[A-Z0-9\-]+', text, re.I):
            return 0.7, 'reference'

        # Check for email
        if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
            return 0.9, 'email'

        # Check for phone
        if re.search(r'(?:\+44|0)\s*\d{2,4}\s*\d{3,4}\s*\d{3,4}', text):
            return 0.75, 'phone'

        return 0.0, None

    def _get_heading_influence(self) -> Optional[str]:
        """Get influence of current heading on content classification"""
        if not self.current_heading:
            return None

        heading_lower = self.current_heading.lower()

        for keyword in self.DYNAMIC_HEADINGS:
            if keyword in heading_lower:
                return 'dynamic'

        for keyword in self.STATIC_HEADINGS:
            if keyword in heading_lower:
                return 'static'

        return None

    def _generate_placeholder_name(self, text: str, content_type: str) -> str:
        """Generate a meaningful placeholder name"""
        # Use heading context if available
        if self.current_heading:
            # Clean heading
            name = re.sub(r'[^a-zA-Z\s]', '', self.current_heading.lower())
            name = '_'.join(name.split()[:3])
            if name:
                return f"{name}_{content_type}"

        # Extract key words from text
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        important_words = [w for w in words[:5] if w not in
                         ['this', 'that', 'with', 'from', 'your', 'have', 'been', 'will', 'would']]

        if important_words:
            return f"{'_'.join(important_words[:2])}_{content_type}"

        return f"section_{content_type}"

    def _generate_prompt_hint(self, text: str) -> str:
        """Generate a hint for LLM prompt based on content"""
        text_lower = text.lower()

        if 'recommend' in text_lower:
            return "Generate personalized investment/financial recommendations based on client circumstances and objectives."
        elif 'assessment' in text_lower or 'risk' in text_lower:
            return "Provide risk assessment and analysis based on client's risk profile and capacity for loss."
        elif 'rationale' in text_lower or 'reason' in text_lower:
            return "Explain the reasoning behind the recommendations, linking to client's specific situation."
        elif 'summary' in text_lower or 'overview' in text_lower:
            return "Provide executive summary of key points and recommendations."
        elif 'circumstance' in text_lower or 'situation' in text_lower:
            return "Describe client's current financial circumstances and objectives."
        elif 'objective' in text_lower or 'goal' in text_lower:
            return "Outline client's financial objectives and goals."
        else:
            return f"Generate appropriate content for this section. Original sample: {text[:100]}..."

    def _detect_template_type(self, full_text: str) -> str:
        """Detect the type of template"""
        text_lower = full_text.lower()

        if 'annual review' in text_lower or 'yearly review' in text_lower:
            return 'annual_review'
        elif 'suitability' in text_lower:
            return 'suitability_report'
        elif 'pension' in text_lower and 'transfer' in text_lower:
            return 'pension_transfer'
        elif 'investment' in text_lower and 'advice' in text_lower:
            return 'investment_advice'
        elif 'protection' in text_lower:
            return 'protection'
        elif 'fact find' in text_lower:
            return 'fact_find'
        else:
            return 'general'

    def _safe_replace(self, content: str, old_text: str, new_text: str) -> str:
        """Safely replace text in XML content"""
        # Escape the new text for XML
        escaped_new = (new_text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))

        return content.replace(old_text, escaped_new)


def create_template_analyzer():
    """Factory function"""
    return TemplateAnalyzer()
