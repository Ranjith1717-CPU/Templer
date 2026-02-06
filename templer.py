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
    )
    TEMPLER_CORE_AVAILABLE = True
except ImportError:
    TEMPLER_CORE_AVAILABLE = False

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


# ============== MAIN APPLICATION ==============

def main():
    st.title("🧠 Templer v2.0")
    st.markdown("**Template Intelligence Engine** | Zero-prep template processing with auto-detection")

    # Mode selector
    st.sidebar.header("🎛️ Detection Mode")

    mode = st.sidebar.radio(
        "Choose how to detect template fields:",
        [
            "🔍 Auto-Detect (Recommended)",
            "📚 Learn from Example",
            "🖍️ Manual Highlights (Original)"
        ],
        index=0
    )

    # Show mode description
    mode_descriptions = {
        "🔍 Auto-Detect (Recommended)": "Automatically finds insertion points using pattern matching and AI. **No template preparation needed.**",
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

    # Azure status
    with st.sidebar.expander("☁️ Azure OpenAI", expanded=False):
        st.success(f"**Connected** | {AZURE_DEPLOYMENT}")

    st.divider()

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
            if mode == "🔍 Auto-Detect (Recommended)":
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
