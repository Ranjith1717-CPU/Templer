"""
Intelligent Template Builder - Streamlit Version
Reads CONTEXT/HEADINGS around highlighted text to understand what data to extract from inputs
"""

import streamlit as st
import re
import json
import io
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple
import requests
import time

# For PDF processing
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Intelligent Template Builder",
    page_icon="🧠",
    layout="wide"
)

# ============== AZURE OPENAI CONFIGURATION ==============
AZURE_API_KEY = st.secrets.get("AZURE_API_KEY", "AHz5fOtxXHg0m0OYl9neAlfOnna79WBhvetPnnZ4nssRlZXiK9FBJQQJ99BIACYeBjFXJ3w3AAABACOGcHNC")
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", "https://curious-01.openai.azure.com/")
AZURE_DEPLOYMENT = st.secrets.get("AZURE_DEPLOYMENT", "Ranjith")
AZURE_API_VERSION = st.secrets.get("AZURE_API_VERSION", "2024-12-01-preview")


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


def extract_text_from_xml(xml_content: str) -> str:
    """Extract plain text from Word XML"""
    text_parts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', xml_content)
    return ' '.join(text_parts)


def detect_highlights_with_context(document_xml: str) -> List[Dict]:
    """
    Detect highlighted text AND the context/heading before it.
    This is crucial - the heading tells us WHAT the highlighted text represents.
    """
    highlights = []

    # First, extract all text with position info to understand document structure
    # Find all paragraphs
    paragraphs = re.findall(r'<w:p[^>]*>(.*?)</w:p>', document_xml, re.DOTALL)

    full_text = extract_text_from_xml(document_xml)

    # Track seen highlights to avoid duplicates
    seen = set()

    # Pattern to find highlighted runs within the XML
    # Look for w:highlight tag and capture surrounding context
    highlight_pattern = r'(<w:p[^>]*>.*?<w:highlight[^>]*/>.*?</w:p>)'

    for para_match in re.finditer(highlight_pattern, document_xml, re.DOTALL):
        para_xml = para_match.group(1)

        # Extract all text from this paragraph
        para_text_parts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', para_xml)
        para_full_text = ''.join(para_text_parts)

        # Find highlighted portions within this paragraph
        # Look for text that comes after highlight tag
        hl_text_pattern = r'<w:highlight\s+w:val="([^"]+)"[^/]*/>\s*</w:rPr>\s*<w:t[^>]*>([^<]+)</w:t>'

        for hl_match in re.finditer(hl_text_pattern, para_xml):
            color = hl_match.group(1)
            highlighted_text = hl_match.group(2).strip()

            if highlighted_text and highlighted_text not in seen:
                seen.add(highlighted_text)

                # Find the context - text BEFORE the highlighted portion in the paragraph
                hl_pos = para_full_text.find(highlighted_text)
                context_before = para_full_text[:hl_pos].strip() if hl_pos > 0 else ""

                # Also try to find context from full document
                doc_pos = full_text.find(highlighted_text)
                if doc_pos > 0:
                    # Get up to 200 chars before
                    wider_context = full_text[max(0, doc_pos-200):doc_pos].strip()
                    # Find last sentence or phrase boundary
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
                    'wider_context': wider_context
                })

    # Also try alternate pattern for different highlight formats
    alt_pattern = r'<w:rPr>(?:[^<]*<[^>]+>)*[^<]*<w:highlight\s+w:val="([^"]+)"[^/]*/>[^<]*(?:<[^>]+>[^<]*)*</w:rPr>\s*<w:t[^>]*>([^<]+)</w:t>'

    for match in re.finditer(alt_pattern, document_xml, re.DOTALL):
        color = match.group(1)
        highlighted_text = match.group(2).strip()

        if highlighted_text and highlighted_text not in seen:
            seen.add(highlighted_text)

            # Get context
            match_pos = match.start()
            context_start = max(0, match_pos - 500)
            context_xml = document_xml[context_start:match_pos]
            context_text_parts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', context_xml)
            context = ' '.join(context_text_parts[-10:])  # Last 10 text elements

            highlights.append({
                'text': highlighted_text,
                'color': color,
                'context': context,
                'wider_context': context
            })

    # Fallback: simple highlight detection
    simple_pattern = r'<w:highlight[^/]*/>.*?<w:t[^>]*>([^<]+)</w:t>'
    for match in re.finditer(simple_pattern, document_xml, re.DOTALL):
        text = match.group(1).strip()
        if text and text not in seen:
            seen.add(text)

            match_pos = match.start()
            context_start = max(0, match_pos - 300)
            context_xml = document_xml[context_start:match_pos]
            context_parts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', context_xml)
            context = ' '.join(context_parts[-5:])

            highlights.append({
                'text': text,
                'color': 'unknown',
                'context': context,
                'wider_context': context
            })

    return highlights


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
    The context tells us WHAT the placeholder represents.
    """

    # Build the placeholders list with context
    placeholders_info = []
    for h in highlights[:60]:  # Limit to 60 for token management
        context = h.get('context', '') or h.get('wider_context', '')
        placeholders_info.append(f"- Context: \"{context}\" → Placeholder: \"{h['text']}\"")

    placeholders_list = "\n".join(placeholders_info)

    # Truncate input intelligently
    max_input = 12000
    if len(input_content) > max_input:
        input_content = input_content[:max_input//2] + "\n...[content truncated]...\n" + input_content[-max_input//2:]

    prompt = f'''I need to fill a template. The template has placeholders (highlighted text) that need to be replaced with actual data from input documents.

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


def generate_filled_document(template_bytes: bytes, mappings: Dict[str, str]) -> bytes:
    """Generate filled Word document"""
    template_io = io.BytesIO(template_bytes)
    output_io = io.BytesIO()

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

                            content = content.replace(original, safe_replacement)

                    data = content.encode('utf-8')

                zout.writestr(item, data)

    return output_io.getvalue()


def main():
    st.title("🧠 Intelligent Template Builder")
    st.markdown("**Upload input data → Upload template → AI reads context & fills placeholders**")

    st.info("💡 **How it works:** The app reads the HEADING/CONTEXT before each highlighted section to understand what data to extract from your input files.")

    # Session state
    if 'processed_doc' not in st.session_state:
        st.session_state.processed_doc = None
    if 'mappings' not in st.session_state:
        st.session_state.mappings = {}
    if 'highlights' not in st.session_state:
        st.session_state.highlights = []

    # API Status
    with st.expander("☁️ Azure OpenAI Configuration", expanded=False):
        st.success(f"**Connected** | Deployment: {AZURE_DEPLOYMENT}")

    # File uploads
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Step 1: Upload Input Data")
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
        st.subheader("📋 Step 2: Upload Template")
        st.caption("Word doc with highlighted placeholders to fill")
        template_file = st.file_uploader(
            "Template",
            type=['docx'],
            key="template",
            label_visibility="collapsed"
        )
        if template_file:
            st.success(f"✅ {template_file.name}")

    st.divider()

    # Process
    can_process = input_files and template_file

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

            # Step 2: Analyze template
            status.text("📋 Analyzing template structure...")
            progress.progress(35)

            _, template_xml, template_bytes = read_word_document(template_file)
            highlights = detect_highlights_with_context(template_xml)
            st.session_state.highlights = highlights

            log(f"Found {len(highlights)} highlighted placeholders")

            if not highlights:
                st.warning("⚠️ No highlighted text found in template!")
                return

            # Show detected placeholders with context
            with st.expander(f"📝 Detected {len(highlights)} Placeholders with Context", expanded=True):
                for h in highlights[:15]:
                    ctx = h.get('context', '')[:50]
                    st.text(f"  Context: \"{ctx}...\" → [{h['text']}]")
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
            log("Applying replacements...")

            filled_doc = generate_filled_document(template_bytes, mappings)
            st.session_state.processed_doc = filled_doc

            progress.progress(100)
            status.text("✅ Done!")
            log("Document ready for download!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            log(f"Error: {e}")

    # Results
    if st.session_state.processed_doc:
        st.divider()
        st.subheader("✅ Document Ready!")

        col1, col2 = st.columns([1, 2])

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
            st.metric("Replacements Made", f"{changed}/{total}")

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
