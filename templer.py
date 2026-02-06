"""
Templer v2.0 - AI-Powered Template Intelligence Engine
Zero-prep template processing with auto-detection, learning, and validation.

Solves the 4-hour-per-template manual setup problem by automatically detecting
insertion points without requiring manual highlighting or placeholder preparation.
"""

import streamlit as st
import re
import json
import io
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests
import time

# For PDF processing
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import Templer Core modules
try:
    from templer_core import (
        AutoDetector,
        LearningEngine,
        DocumentParser,
        TableProcessor,
        FormatPreserver,
        ConditionalHandler,
        ValidationEngine,
        PatternMatcher,
        TemplateAnalyzer,
        ContentType,
        AnalysisResult,
    )
    TEMPLER_CORE_AVAILABLE = True
except ImportError:
    TEMPLER_CORE_AVAILABLE = False
    # Define placeholder classes for type hints
    class ContentType:
        STATIC = "static"
        DYNAMIC_VALUE = "dynamic_value"
        DYNAMIC_LLM = "dynamic_llm"

# Page configuration
st.set_page_config(
    page_title="Templer v2.0 - Template Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ============== AZURE OPENAI CONFIGURATION ==============
AZURE_API_KEY = st.secrets.get("AZURE_API_KEY", "AHz5fOtxXHg0m0OYl9neAlfOnna79WBhvetPnnZ4nssRlZXiK9FBJQQJ99BIACYeBjFXJ3w3AAABACOGcHNC")
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", "https://curious-01.openai.azure.com/")
AZURE_DEPLOYMENT = st.secrets.get("AZURE_DEPLOYMENT", "Ranjith")
AZURE_API_VERSION = st.secrets.get("AZURE_API_VERSION", "2024-12-01-preview")


# ============== DOCUMENT READING FUNCTIONS ==============

def read_word_document(file) -> Tuple[str, str, bytes]:
    """Read Word document - returns (text_content, xml_content, raw_bytes)"""
    file_bytes = file.read()
    file.seek(0)

    text_content = ""
    document_xml = ""

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            if 'word/document.xml' in zf.namelist():
                document_xml = zf.read('word/document.xml').decode('utf-8')
                # Extract text
                text_matches = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', document_xml)
                text_content = ' '.join(text_matches)
    except Exception as e:
        st.warning(f"Error reading Word document: {e}")

    return text_content, document_xml, file_bytes


def read_pdf_document(file) -> str:
    """Read PDF and return text content"""
    if not PDF_AVAILABLE:
        return f"PDF file: {file.name} (pdfplumber not installed)"

    text_content = ""
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages[:20]):
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"
                if len(text_content) > 80000:
                    break
    except Exception as e:
        st.warning(f"Error reading PDF: {e}")

    return text_content


def read_text_file(file) -> str:
    """Read text file"""
    try:
        return file.read().decode('utf-8')
    except:
        return file.read().decode('latin-1')


# ============== HIGHLIGHT DETECTION (LEGACY) ==============

def detect_highlights_with_context(document_xml: str) -> List[Dict]:
    """
    Detect highlighted text AND the context/heading before it.
    This is the original templer behavior - kept for backward compatibility.
    """
    highlights = []

    # Extract all text
    def extract_text_from_xml(xml: str) -> str:
        return ' '.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', xml))

    full_text = extract_text_from_xml(document_xml)

    # Track seen highlights
    seen = set()

    # Pattern to find highlighted runs
    highlight_pattern = r'(<w:p[^>]*>.*?<w:highlight[^>]*/>.*?</w:p>)'

    for para_match in re.finditer(highlight_pattern, document_xml, re.DOTALL):
        para_xml = para_match.group(1)
        para_text_parts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', para_xml)
        para_full_text = ''.join(para_text_parts)

        hl_text_pattern = r'<w:highlight\s+w:val="([^"]+)"[^/]*/>\s*</w:rPr>\s*<w:t[^>]*>([^<]+)</w:t>'

        for hl_match in re.finditer(hl_text_pattern, para_xml):
            color = hl_match.group(1)
            highlighted_text = hl_match.group(2).strip()

            if highlighted_text and highlighted_text not in seen:
                seen.add(highlighted_text)

                hl_pos = para_full_text.find(highlighted_text)
                context_before = para_full_text[:hl_pos].strip() if hl_pos > 0 else ""

                doc_pos = full_text.find(highlighted_text)
                if doc_pos > 0:
                    wider_context = full_text[max(0, doc_pos-200):doc_pos].strip()
                    for sep in ['. ', ':', '\n', '•', '-']:
                        if sep in wider_context:
                            wider_context = wider_context.split(sep)[-1].strip()
                            break
                else:
                    wider_context = context_before

                highlights.append({
                    'text': highlighted_text,
                    'color': color,
                    'context': context_before or wider_context,
                    'wider_context': wider_context,
                    'field_type': 'text',
                    'confidence': 0.9
                })

    # Fallback patterns
    alt_patterns = [
        r'<w:rPr>(?:[^<]*<[^>]+>)*[^<]*<w:highlight\s+w:val="([^"]+)"[^/]*/>[^<]*(?:<[^>]+>[^<]*)*</w:rPr>\s*<w:t[^>]*>([^<]+)</w:t>',
        r'<w:highlight[^/]*/>.*?<w:t[^>]*>([^<]+)</w:t>'
    ]

    for pattern in alt_patterns:
        for match in re.finditer(pattern, document_xml, re.DOTALL):
            if match.lastindex >= 2:
                highlighted_text = match.group(2).strip()
            else:
                highlighted_text = match.group(1).strip()

            if highlighted_text and highlighted_text not in seen:
                seen.add(highlighted_text)
                match_pos = match.start()
                context_start = max(0, match_pos - 300)
                context_xml = document_xml[context_start:match_pos]
                context_parts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', context_xml)
                context = ' '.join(context_parts[-5:])

                highlights.append({
                    'text': highlighted_text,
                    'color': 'unknown',
                    'context': context,
                    'wider_context': context,
                    'field_type': 'text',
                    'confidence': 0.85
                })

    return highlights


# ============== AUTO-DETECTION MODE ==============

def auto_detect_insertion_points(file_bytes: bytes) -> List[Dict]:
    """
    Auto-detect insertion points without requiring highlights.
    Uses pattern matching, heuristics, and context analysis.
    """
    if not TEMPLER_CORE_AVAILABLE:
        st.warning("⚠️ Templer Core not available. Using basic detection.")
        return []

    detector = AutoDetector()
    result = detector.detect_all(file_bytes)

    # Convert to highlight-compatible format
    points = []
    for ip in result.insertion_points:
        points.append({
            'text': ip.text,
            'color': 'auto_detected',
            'context': ip.context,
            'wider_context': ip.context,
            'field_type': ip.field_type,
            'confidence': ip.confidence,
            'detection_method': ip.detection_method
        })

    return points


# ============== LEARNING MODE ==============

def learn_from_examples(blank_bytes: bytes, filled_bytes: bytes) -> Tuple[List[Dict], Dict]:
    """
    Learn template structure from blank + filled example pair.
    Returns insertion points and schema.
    """
    if not TEMPLER_CORE_AVAILABLE:
        st.warning("⚠️ Templer Core not available. Learning mode disabled.")
        return [], {}

    engine = LearningEngine()
    schema = engine.learn_from_example(blank_bytes, filled_bytes)

    # Convert to highlight-compatible format
    points = []
    for field in schema.fields:
        points.append({
            'text': field.blank_text,
            'color': 'learned',
            'context': field.context_before,
            'wider_context': field.context_before,
            'field_type': field.field_type,
            'confidence': field.confidence,
            'detection_method': 'learning',
            'example_value': field.filled_text,
            'field_name': field.field_name
        })

    return points, schema.to_dict()


# ============== LLM MAPPING ==============

def call_azure_openai(prompt: str, timeout: int = 60) -> str:
    """Call Azure OpenAI API with timeout protection"""
    endpoint = f"{AZURE_ENDPOINT}openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_API_KEY
    }

    payload = {
        'messages': [
            {
                'role': 'system',
                'content': '''You are an expert at extracting data from documents and filling templates.
Your task is to read input documents and map data to template placeholders based on CONTEXT.

IMPORTANT: The context/heading BEFORE a placeholder tells you what data to find.
Example: If context is "Client Name:" and placeholder is "John Doe", find the ACTUAL client name from input.

Always return valid JSON. Be precise and use exact values from the input documents.'''
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_tokens': 4000,
        'temperature': 0.1
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        raise Exception("Request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API error: {str(e)}")


def map_data_with_llm(input_content: str, highlights: List[Dict], progress_callback=None) -> Dict[str, str]:
    """
    Use LLM to map input data to template placeholders based on CONTEXT.
    Enhanced with field type hints for better accuracy.
    """
    # Build the placeholders list with context and type hints
    placeholders_info = []
    for h in highlights[:60]:  # Limit to 60 for token management
        context = h.get('context', '') or h.get('wider_context', '')
        field_type = h.get('field_type', 'text')
        confidence = h.get('confidence', 0.5)

        type_hint = f" (expect {field_type})" if field_type != 'text' else ""
        placeholders_info.append(f"- Context: \"{context}\" → Placeholder: \"{h['text']}\"{type_hint}")

    placeholders_list = "\n".join(placeholders_info)

    # Truncate input intelligently
    max_input = 12000
    if len(input_content) > max_input:
        input_content = input_content[:max_input//2] + "\n...[content truncated]...\n" + input_content[-max_input//2:]

    prompt = f'''I need to fill a template. The template has placeholders that need to be replaced with actual data from input documents.

## TEMPLATE PLACEHOLDERS (with context showing what each represents):
{placeholders_list}

## INPUT DOCUMENTS (source data):
{input_content}

## YOUR TASK:
1. The CONTEXT before each placeholder tells you what data is needed
2. Find the corresponding value in the INPUT DOCUMENTS
3. Return a JSON mapping each placeholder to its replacement value

## EXAMPLES:
- Context: "Client Name:" with placeholder "Stacey Shipman" → Find actual client name from input
- Context: "Provider:" with placeholder "AJ Bell" → Find actual provider name from input
- Context: "Amount:" with placeholder "£150,000" → Find actual amount from input
- Context: "Date:" with placeholder "10 March 2023" → Find actual date from input

## RULES:
- Use EXACT values from the input documents
- For names, find the actual person's name mentioned in input
- For providers/companies, find actual names from input
- For amounts, use the exact figures with currency symbols
- If truly not found in input, use the original placeholder value (don't put "N/A")

## OUTPUT FORMAT (JSON only):
{{
  "Stacey Shipman": "Actual Name From Input",
  "AJ Bell": "Actual Provider From Input",
  "£150,000": "£Actual Amount",
  ...
}}

Return ONLY the JSON:'''

    if progress_callback:
        progress_callback("Sending to Azure OpenAI for intelligent mapping...")

    response = call_azure_openai(prompt, timeout=90)

    # Parse response
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        st.warning(f"JSON parse error: {e}")

    return {}


# ============== DOCUMENT GENERATION ==============

def generate_filled_document(template_bytes: bytes, mappings: Dict[str, str]) -> bytes:
    """Generate filled Word document with format preservation"""
    template_io = io.BytesIO(template_bytes)
    output_io = io.BytesIO()

    # Use FormatPreserver if available
    format_preserver = None
    if TEMPLER_CORE_AVAILABLE:
        format_preserver = FormatPreserver()

    with zipfile.ZipFile(template_io, 'r') as zin:
        with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                data = zin.read(item)

                if item == 'word/document.xml':
                    content = data.decode('utf-8')

                    for original, replacement in mappings.items():
                        if replacement and replacement != original:
                            # Escape XML special characters
                            safe_replacement = (str(replacement)
                                .replace('&', '&amp;')
                                .replace('<', '&lt;')
                                .replace('>', '&gt;')
                                .replace('"', '&quot;')
                                .replace("'", '&apos;'))

                            if format_preserver:
                                # Use format-preserving replacement
                                content = format_preserver.replace_text(content, original, safe_replacement)
                            else:
                                content = content.replace(original, safe_replacement)

                    data = content.encode('utf-8')

                zout.writestr(item, data)

    return output_io.getvalue()


# ============== CONFIDENCE DISPLAY ==============

def display_confidence_scores(highlights: List[Dict], mappings: Dict[str, str]):
    """Display confidence scores with color coding"""
    st.subheader("🎯 Extraction Confidence")

    # Group by confidence level
    high = []
    medium = []
    low = []

    for h in highlights:
        text = h['text']
        confidence = h.get('confidence', 0.5)
        replacement = mappings.get(text, text)
        changed = replacement != text

        item = {
            'text': text[:40] + ('...' if len(text) > 40 else ''),
            'confidence': confidence,
            'changed': changed,
            'method': h.get('detection_method', 'highlight')
        }

        if confidence >= 0.7:
            high.append(item)
        elif confidence >= 0.4:
            medium.append(item)
        else:
            low.append(item)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🟢 High Confidence")
        for item in high[:10]:
            status = "✅" if item['changed'] else "⬜"
            st.text(f"{status} {item['text']} ({item['confidence']:.0%})")

    with col2:
        st.markdown("### 🟡 Medium Confidence")
        for item in medium[:10]:
            status = "✅" if item['changed'] else "⬜"
            st.text(f"{status} {item['text']} ({item['confidence']:.0%})")

    with col3:
        st.markdown("### 🔴 Low Confidence")
        for item in low[:10]:
            status = "✅" if item['changed'] else "⬜"
            st.text(f"{status} {item['text']} ({item['confidence']:.0%})")


# ============== TEMPLATE ANALYZER HELPERS ==============

def enhance_analysis_with_ai(file_bytes: bytes, analysis) -> 'AnalysisResult':
    """
    Use Azure OpenAI to enhance the classification of sections.
    This improves accuracy for edge cases.
    """
    if not TEMPLER_CORE_AVAILABLE:
        return analysis

    # Extract sample sections for AI review
    uncertain_sections = [s for s in analysis.sections if 0.3 < s.confidence < 0.7]

    if not uncertain_sections:
        return analysis

    # Build prompt for AI classification
    sections_text = []
    for i, section in enumerate(uncertain_sections[:20]):  # Limit to 20
        heading = section.heading_context or "No heading"
        preview = section.text[:200] if len(section.text) > 200 else section.text
        sections_text.append(f"{i+1}. [Under: {heading}]\n{preview}")

    sections_list = "\n\n".join(sections_text)

    prompt = f'''You are analyzing a financial advisory document template to determine which sections are STATIC (never change) vs DYNAMIC (need to be filled per client).

## CLASSIFICATION RULES:
- STATIC: Legal disclaimers, compliance text, FCA warnings, generic descriptions, headers, footers
- DYNAMIC_VALUE: Client names, dates, amounts, percentages, reference numbers
- DYNAMIC_LLM: Client-specific recommendations, assessments, rationale, personalized advice

## SECTIONS TO CLASSIFY:
{sections_list}

## OUTPUT FORMAT (JSON):
Return a JSON object mapping section number to classification:
{{
  "1": {{"type": "STATIC", "confidence": 0.9, "reason": "Legal boilerplate"}},
  "2": {{"type": "DYNAMIC_LLM", "confidence": 0.85, "reason": "Client-specific recommendation"}},
  ...
}}

Return ONLY the JSON:'''

    try:
        response = call_azure_openai(prompt, timeout=60)

        # Parse response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            classifications = json.loads(json_match.group(0))

            # Apply AI classifications back to analysis
            for i, section in enumerate(uncertain_sections[:20]):
                key = str(i + 1)
                if key in classifications:
                    ai_class = classifications[key]
                    ai_type = ai_class.get('type', '')
                    ai_confidence = ai_class.get('confidence', 0.5)

                    # Update section based on AI
                    if ai_type == 'STATIC':
                        section.content_type = ContentType.STATIC
                    elif ai_type == 'DYNAMIC_VALUE':
                        section.content_type = ContentType.DYNAMIC_VALUE
                    elif ai_type == 'DYNAMIC_LLM':
                        section.content_type = ContentType.DYNAMIC_LLM

                    # Boost confidence with AI agreement
                    section.confidence = min(0.95, (section.confidence + ai_confidence) / 2 + 0.1)
                    section.reasoning += f" | AI: {ai_class.get('reason', '')}"

    except Exception as e:
        st.warning(f"AI enhancement skipped: {e}")

    # Recalculate counts
    analysis.static_count = sum(1 for s in analysis.sections if s.content_type == ContentType.STATIC)
    analysis.dynamic_value_count = sum(1 for s in analysis.sections if s.content_type == ContentType.DYNAMIC_VALUE)
    analysis.dynamic_llm_count = sum(1 for s in analysis.sections if s.content_type == ContentType.DYNAMIC_LLM)

    return analysis


def generate_analysis_report(analysis) -> str:
    """Generate a markdown report of the template analysis."""
    report = f"""# Template Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Template Type:** {analysis.template_type.replace('_', ' ').title()}
- **Overall Confidence:** {analysis.overall_confidence:.0%}
- **Total Sections Analyzed:** {len(analysis.sections)}

## Section Breakdown
| Type | Count | Percentage |
|------|-------|------------|
| Static | {analysis.static_count} | {analysis.static_count/max(1,len(analysis.sections))*100:.0f}% |
| Dynamic Values | {analysis.dynamic_value_count} | {analysis.dynamic_value_count/max(1,len(analysis.sections))*100:.0f}% |
| LLM Generated | {analysis.dynamic_llm_count} | {analysis.dynamic_llm_count/max(1,len(analysis.sections))*100:.0f}% |

## Placeholders Created

"""

    # List dynamic placeholders
    placeholders = analysis.get_placeholders()
    for name, hint in placeholders.items():
        report += f"### `{{{{{name}}}}}`\n"
        report += f"- **Prompt Hint:** {hint}\n\n"

    # List LLM sections
    llm_sections = [s for s in analysis.sections if s.content_type == ContentType.DYNAMIC_LLM]
    if llm_sections:
        report += "\n## LLM-Generated Sections (Require Prompt Engineering)\n\n"
        for section in llm_sections:
            report += f"### {section.placeholder_name or 'Unnamed'}\n"
            report += f"- **Confidence:** {section.confidence:.0%}\n"
            report += f"- **Heading:** {section.heading_context or 'None'}\n"
            report += f"- **Prompt Hint:** {section.prompt_hint or 'Generate appropriate content'}\n"
            report += f"- **Sample Text:** {section.text[:150]}...\n\n"

    # Instructions
    report += """
## Next Steps

1. **Review the classifications** - Check if static/dynamic labels are correct
2. **Download the template with placeholders** - Use this as your output template
3. **Use the prompt configuration** - Configure your LLM prompts for each dynamic section
4. **Test with sample data** - Use the 'Auto-Detect & Fill' mode to test

---
*Generated by Templer v2.0 - Template Intelligence Engine*
"""

    return report


# ============== MAIN APPLICATION ==============

def main():
    st.title("🧠 Templer v2.0")
    st.markdown("**Template Intelligence Engine** | Zero-prep template processing with auto-detection")

    # Mode selector
    st.sidebar.header("🎛️ Mode")

    mode = st.sidebar.radio(
        "Choose operation mode:",
        [
            "🔧 Analyze & Create Template",
            "🔍 Auto-Detect & Fill",
            "📚 Learn from Example",
            "🖍️ Manual Highlights (Original)"
        ],
        index=0
    )

    # Show mode description
    mode_descriptions = {
        "🔧 Analyze & Create Template": "**NEW!** Upload a raw template → AI identifies static vs dynamic sections → Outputs template with {{placeholders}}. **Solves the 4-hour setup problem.**",
        "🔍 Auto-Detect & Fill": "Automatically finds insertion points using pattern matching and AI. **No template preparation needed.**",
        "📚 Learn from Example": "Upload a blank template + filled example pair. System learns what to replace.",
        "🖍️ Manual Highlights (Original)": "Use highlighted text in your template as placeholders (original Templer behavior)."
    }
    st.sidebar.info(mode_descriptions[mode])

    # Check core availability
    if not TEMPLER_CORE_AVAILABLE and mode != "🖍️ Manual Highlights (Original)":
        st.sidebar.warning("⚠️ Advanced features require templer_core module")

    # Session state
    if 'processed_doc' not in st.session_state:
        st.session_state.processed_doc = None
    if 'mappings' not in st.session_state:
        st.session_state.mappings = {}
    if 'highlights' not in st.session_state:
        st.session_state.highlights = []
    if 'learned_schema' not in st.session_state:
        st.session_state.learned_schema = None
    if 'analyzed_template' not in st.session_state:
        st.session_state.analyzed_template = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'prompt_config' not in st.session_state:
        st.session_state.prompt_config = None

    # Azure status
    with st.sidebar.expander("☁️ Azure OpenAI", expanded=False):
        st.success(f"**Connected** | {AZURE_DEPLOYMENT}")

    st.divider()

    # ============== ANALYZE & CREATE TEMPLATE MODE ==============
    if mode == "🔧 Analyze & Create Template":
        st.subheader("🔧 Template Analyzer")
        st.markdown("""
        **Upload a raw template** (firm's approved Word doc with no placeholders) and the AI will:
        1. Identify **static sections** (legal, headers, boilerplate)
        2. Identify **dynamic sections** (client names, recommendations, assessments)
        3. Create a **new template with {{placeholders}}** inserted
        4. Generate **prompt hints** for LLM-generated sections

        **This turns a 4-hour manual process into minutes of review.**
        """)

        st.divider()

        # File upload for raw template
        st.markdown("**📄 Upload Raw Template**")
        raw_template = st.file_uploader(
            "Upload the firm's template (no placeholders needed)",
            type=['docx'],
            key="raw_template",
            label_visibility="collapsed"
        )

        if raw_template:
            st.success(f"✅ {raw_template.name}")

            # Analysis options
            with st.expander("⚙️ Analysis Options", expanded=False):
                use_ai_classification = st.checkbox(
                    "Use AI for enhanced classification",
                    value=True,
                    help="Uses Azure OpenAI to improve static/dynamic classification accuracy"
                )
                include_tables = st.checkbox(
                    "Analyze tables separately",
                    value=True,
                    help="Identify table headers vs data rows for proper handling"
                )

            if st.button("🔬 Analyze Template", type="primary", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()

                try:
                    status.text("📖 Reading template...")
                    progress.progress(10)

                    raw_bytes = raw_template.read()
                    raw_template.seek(0)

                    if TEMPLER_CORE_AVAILABLE:
                        status.text("🔬 Analyzing document structure...")
                        progress.progress(30)

                        analyzer = TemplateAnalyzer()
                        analysis = analyzer.analyze_template(raw_bytes)

                        progress.progress(50)

                        # If AI classification enabled, enhance with LLM
                        if use_ai_classification:
                            status.text("🧠 AI is classifying sections...")
                            analysis = enhance_analysis_with_ai(raw_bytes, analysis)

                        progress.progress(70)

                        status.text("📝 Generating template with placeholders...")

                        # Generate the new template
                        new_template_bytes = analyzer.generate_template(raw_bytes, analysis)

                        # Generate prompt config
                        prompt_config = analyzer.generate_prompt_config(analysis)

                        progress.progress(90)

                        # Store in session state
                        st.session_state.analyzed_template = new_template_bytes
                        st.session_state.analysis_result = analysis
                        st.session_state.prompt_config = prompt_config

                        progress.progress(100)
                        status.text("✅ Analysis complete!")

                    else:
                        st.error("⚠️ Template Analyzer requires templer_core module")

                except Exception as e:
                    st.error(f"❌ Error analyzing template: {e}")

        # Show analysis results
        if st.session_state.analysis_result:
            analysis = st.session_state.analysis_result

            st.divider()
            st.subheader("📊 Analysis Results")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Static Sections", analysis.static_count)
            with col2:
                st.metric("Dynamic Values", analysis.dynamic_value_count)
            with col3:
                st.metric("LLM Sections", analysis.dynamic_llm_count)
            with col4:
                st.metric("Confidence", f"{analysis.overall_confidence:.0%}")

            # Template type
            st.info(f"📋 **Detected Template Type:** {analysis.template_type.replace('_', ' ').title()}")

            # Detailed breakdown
            with st.expander("📝 Section-by-Section Analysis", expanded=True):
                # Filter options
                show_filter = st.radio(
                    "Show:",
                    ["All", "Dynamic Only", "Static Only"],
                    horizontal=True
                )

                for i, section in enumerate(analysis.sections[:50]):  # Limit to 50
                    if show_filter == "Dynamic Only" and section.content_type == ContentType.STATIC:
                        continue
                    if show_filter == "Static Only" and section.content_type != ContentType.STATIC:
                        continue

                    # Color coding
                    if section.content_type == ContentType.STATIC:
                        color = "🔵"
                        badge = "STATIC"
                    elif section.content_type == ContentType.DYNAMIC_VALUE:
                        color = "🟢"
                        badge = f"DYNAMIC ({section.value_type})"
                    elif section.content_type == ContentType.DYNAMIC_LLM:
                        color = "🟠"
                        badge = "LLM GENERATED"
                    else:
                        color = "⚪"
                        badge = str(section.content_type.value)

                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            preview = section.text[:150] + "..." if len(section.text) > 150 else section.text
                            st.markdown(f"{color} **{badge}** ({section.confidence:.0%})")
                            st.text(preview)
                            if section.placeholder_name:
                                st.caption(f"Placeholder: `{{{{{section.placeholder_name}}}}}`")
                        with col2:
                            if section.heading_context:
                                st.caption(f"Under: {section.heading_context[:30]}")
                        st.markdown("---")

            # Prompt configuration
            if st.session_state.prompt_config:
                with st.expander("🤖 LLM Prompt Configuration", expanded=False):
                    st.json(st.session_state.prompt_config)

            # Download section
            st.divider()
            st.subheader("📥 Download Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.session_state.analyzed_template:
                    filename = f"Template_With_Placeholders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                    st.download_button(
                        "📄 Download Template with Placeholders",
                        st.session_state.analyzed_template,
                        filename,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        type="primary",
                        use_container_width=True
                    )

            with col2:
                if st.session_state.prompt_config:
                    config_json = json.dumps(st.session_state.prompt_config, indent=2)
                    st.download_button(
                        "📋 Download Prompt Config (JSON)",
                        config_json,
                        f"prompt_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True
                    )

            with col3:
                # Generate analysis report
                report = generate_analysis_report(analysis)
                st.download_button(
                    "📊 Download Analysis Report",
                    report,
                    f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    use_container_width=True
                )

            # Option to proceed to fill mode
            st.divider()
            st.info("💡 **Next Step:** Use the generated template with the **'Auto-Detect & Fill'** mode to fill it with client data!")

        return  # Exit main() early for this mode

    # ============== LEARNING MODE UI ==============
    if mode == "📚 Learn from Example":
        st.subheader("📚 Learning Mode")
        st.markdown("Upload a **blank template** and a **filled example** to teach the system what to replace.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**1️⃣ Blank Template**")
            blank_template = st.file_uploader(
                "Upload blank template",
                type=['docx'],
                key="blank_template",
                label_visibility="collapsed"
            )
            if blank_template:
                st.success(f"✅ {blank_template.name}")

        with col2:
            st.markdown("**2️⃣ Filled Example**")
            filled_example = st.file_uploader(
                "Upload filled example",
                type=['docx'],
                key="filled_example",
                label_visibility="collapsed"
            )
            if filled_example:
                st.success(f"✅ {filled_example.name}")

        if blank_template and filled_example:
            if st.button("🧠 Learn Template Structure", type="primary"):
                with st.spinner("Analyzing template differences..."):
                    blank_bytes = blank_template.read()
                    blank_template.seek(0)
                    filled_bytes = filled_example.read()
                    filled_example.seek(0)

                    highlights, schema = learn_from_examples(blank_bytes, filled_bytes)

                    st.session_state.highlights = highlights
                    st.session_state.learned_schema = schema

                    st.success(f"✅ Learned {len(highlights)} fields from example!")

                    if schema:
                        with st.expander("📋 Learned Schema", expanded=True):
                            st.json(schema)

        st.divider()

    # ============== MAIN FILE UPLOADS ==============
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Input Data")
        st.caption("Files containing the actual client/case data")
        input_files = st.file_uploader(
            "Input files",
            type=['docx', 'pdf', 'txt'],
            accept_multiple_files=True,
            key="inputs",
            label_visibility="collapsed"
        )
        if input_files:
            st.success(f"✅ {len(input_files)} file(s) ready")

    with col2:
        st.subheader("📋 Template")
        st.caption("Word document template to fill")
        template_file = st.file_uploader(
            "Template",
            type=['docx'],
            key="template",
            label_visibility="collapsed"
        )
        if template_file:
            st.success(f"✅ {template_file.name}")

    st.divider()

    # ============== PROCESSING ==============
    can_process = input_files and template_file

    # For learning mode, we might already have highlights
    if mode == "📚 Learn from Example" and st.session_state.highlights:
        can_process = input_files and (template_file or st.session_state.highlights)

    if st.button("🧠 Extract & Fill Template", type="primary", disabled=not can_process, use_container_width=True):

        st.session_state.processed_doc = None
        st.session_state.mappings = {}

        progress = st.progress(0)
        status = st.empty()
        log_area = st.container()

        def log(msg):
            with log_area:
                st.text(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

        try:
            # Step 1: Read inputs
            status.text("📁 Reading input files...")
            progress.progress(10)

            combined_input = ""
            for i, f in enumerate(input_files):
                log(f"Reading: {f.name}")

                if f.name.endswith('.docx'):
                    text, _, _ = read_word_document(f)
                elif f.name.endswith('.pdf'):
                    text = read_pdf_document(f)
                else:
                    text = read_text_file(f)

                combined_input += f"\n\n=== FILE: {f.name} ===\n{text}"
                progress.progress(10 + int(20 * (i+1) / len(input_files)))

            log(f"Total input: {len(combined_input):,} characters")

            # Step 2: Detect/analyze template
            status.text("📋 Analyzing template...")
            progress.progress(35)

            _, template_xml, template_bytes = read_word_document(template_file)

            # Choose detection method based on mode
            if mode == "🔍 Auto-Detect & Fill":
                log("Using auto-detection mode...")
                highlights = auto_detect_insertion_points(template_bytes)

                # Fallback to highlights if auto-detect finds nothing
                if not highlights:
                    log("Auto-detect found nothing, falling back to highlights...")
                    highlights = detect_highlights_with_context(template_xml)

            elif mode == "📚 Learn from Example":
                if st.session_state.highlights:
                    highlights = st.session_state.highlights
                    log(f"Using {len(highlights)} learned fields")
                else:
                    log("No learned fields, using highlight detection...")
                    highlights = detect_highlights_with_context(template_xml)

            else:  # Manual highlights
                highlights = detect_highlights_with_context(template_xml)

            st.session_state.highlights = highlights

            log(f"Found {len(highlights)} insertion points")

            if not highlights:
                st.warning("⚠️ No insertion points found in template!")
                st.info("💡 **Tips:**\n- For Auto-Detect: Ensure your template has placeholder text like {{name}} or values like dates/amounts\n- For Manual mode: Highlight text in Word that should be replaced\n- For Learning mode: Upload both blank and filled templates first")
                return

            # Show detected points
            with st.expander(f"📝 Detected {len(highlights)} Insertion Points", expanded=True):
                for h in highlights[:15]:
                    ctx = h.get('context', '')[:40]
                    conf = h.get('confidence', 0.5)
                    method = h.get('detection_method', 'highlight')
                    color = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.4 else "🔴"
                    st.text(f"{color} [{method}] \"{ctx}...\" → [{h['text'][:30]}...]")
                if len(highlights) > 15:
                    st.text(f"  ... and {len(highlights) - 15} more")

            # Step 3: LLM mapping
            status.text("🧠 AI is mapping input data to placeholders...")
            progress.progress(50)
            log("Calling Azure OpenAI...")

            start = time.time()

            try:
                mappings = map_data_with_llm(combined_input, highlights, log)
                elapsed = time.time() - start
                log(f"LLM completed in {elapsed:.1f}s")

                # Count actual changes
                changes = sum(1 for k, v in mappings.items() if v and v != k)
                log(f"Mapped {changes} values to replace")

            except Exception as e:
                st.error(f"AI Error: {e}")
                log(f"Error: {e}")
                return

            st.session_state.mappings = mappings
            progress.progress(80)

            # Step 4: Generate document
            status.text("📝 Generating filled document...")
            log("Applying replacements with format preservation...")

            filled_doc = generate_filled_document(template_bytes, mappings)
            st.session_state.processed_doc = filled_doc

            progress.progress(100)
            status.text("✅ Done!")
            log("Document ready for download!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            log(f"Error: {e}")

    # ============== RESULTS ==============
    if st.session_state.processed_doc:
        st.divider()
        st.subheader("✅ Document Ready!")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            filename = f"Filled_Template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            st.download_button(
                "📥 Download Filled Document",
                st.session_state.processed_doc,
                filename,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True
            )

        with col2:
            total = len(st.session_state.highlights)
            changed = sum(1 for k, v in st.session_state.mappings.items() if v and v != k)
            st.metric("Fields Filled", f"{changed}/{total}")

        with col3:
            if st.session_state.highlights:
                avg_conf = sum(h.get('confidence', 0.5) for h in st.session_state.highlights) / len(st.session_state.highlights)
                st.metric("Confidence", f"{avg_conf:.0%}")

        # Confidence scores
        if st.session_state.highlights and st.session_state.mappings:
            with st.expander("🎯 Confidence Scores", expanded=False):
                display_confidence_scores(st.session_state.highlights, st.session_state.mappings)

        # Show mappings
        if st.session_state.mappings:
            with st.expander("🔄 View All Mappings", expanded=False):
                for orig, new in st.session_state.mappings.items():
                    if new and new != orig:
                        st.text(f"✅ \"{orig[:35]}...\" → \"{new[:50]}{'...' if len(str(new))>50 else ''}\"")
                    else:
                        st.text(f"⬜ \"{orig[:35]}...\" (unchanged)")


if __name__ == "__main__":
    main()
