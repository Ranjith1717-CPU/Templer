"""
Intelligent Template Builder - Streamlit Version
Extracts data from input files and fills highlighted sections in Word templates using Azure OpenAI
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
            for i, page in enumerate(pdf.pages[:20]):  # Limit pages
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"
                if len(text_content) > 80000:
                    break
    except Exception as e:
        st.warning(f"Error reading PDF: {e}")
        text_content = f"PDF file: {file.name}"

    return text_content


def read_text_file(file) -> str:
    """Read text file"""
    try:
        return file.read().decode('utf-8')
    except:
        return file.read().decode('latin-1')


def detect_all_highlights(document_xml: str) -> List[Dict]:
    """
    Detect ALL highlighted sections in Word document.
    Handles yellow, blue, cyan, green, and any other highlight colors.
    """
    highlights = []
    seen_texts = set()

    # Pattern 1: Direct highlight tag with any color
    # <w:highlight w:val="yellow"/> or <w:highlight w:val="cyan"/> etc.
    pattern1 = r'<w:highlight\s+w:val="([^"]+)"[^>]*/>\s*</w:rPr>\s*<w:t[^>]*>([^<]+)</w:t>'
    for match in re.finditer(pattern1, document_xml, re.IGNORECASE):
        color = match.group(1)
        text = match.group(2).strip()
        if text and text not in seen_texts and len(text) > 0:
            seen_texts.add(text)
            highlights.append({
                'text': text,
                'color': color,
                'type': 'highlight'
            })

    # Pattern 2: Highlight within run properties (more complex nesting)
    pattern2 = r'<w:rPr>[^<]*(?:<[^>]+>[^<]*)*<w:highlight\s+w:val="([^"]+)"[^>]*/>[^<]*(?:<[^>]+>[^<]*)*</w:rPr>\s*<w:t[^>]*>([^<]+)</w:t>'
    for match in re.finditer(pattern2, document_xml, re.IGNORECASE | re.DOTALL):
        color = match.group(1)
        text = match.group(2).strip()
        if text and text not in seen_texts and len(text) > 0:
            seen_texts.add(text)
            highlights.append({
                'text': text,
                'color': color,
                'type': 'highlight'
            })

    # Pattern 3: Shading (sometimes used instead of highlight)
    pattern3 = r'<w:shd\s+[^>]*w:fill="([^"]+)"[^>]*/>[^<]*(?:<[^>]+>[^<]*)*</w:rPr>\s*<w:t[^>]*>([^<]+)</w:t>'
    for match in re.finditer(pattern3, document_xml, re.IGNORECASE | re.DOTALL):
        color = match.group(1)
        text = match.group(2).strip()
        # Skip white/auto colors
        if color.upper() not in ['FFFFFF', 'AUTO', 'NONE'] and text and text not in seen_texts:
            seen_texts.add(text)
            highlights.append({
                'text': text,
                'color': color,
                'type': 'shading'
            })

    # Pattern 4: Simple highlight detection (fallback)
    pattern4 = r'<w:highlight[^/]*/>.*?<w:t[^>]*>([^<]+)</w:t>'
    for match in re.finditer(pattern4, document_xml, re.IGNORECASE | re.DOTALL):
        text = match.group(1).strip()
        if text and text not in seen_texts and len(text) > 0:
            seen_texts.add(text)
            highlights.append({
                'text': text,
                'color': 'unknown',
                'type': 'highlight'
            })

    return highlights


def call_azure_openai(prompt: str, timeout: int = 45) -> str:
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
                'content': 'You are an expert at extracting and mapping data from documents. Always respond with valid JSON only. Be precise and accurate.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_tokens': 4000,
        'temperature': 0.1  # Low temperature for accuracy
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        raise Exception("Request timed out - try with smaller input")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API error: {str(e)}")


def extract_and_map_with_llm(input_content: str, highlights: List[Dict], progress_callback=None) -> Dict[str, str]:
    """
    Use LLM to extract data from input and map to template highlights.
    This is the CORE intelligence - LLM sees both input data and what needs to be filled.
    """

    # Prepare the list of placeholders to fill
    placeholders_list = "\n".join([f"- \"{h['text']}\"" for h in highlights[:50]])  # Limit to 50

    # Truncate input to prevent timeout (keep most important parts)
    max_input_length = 15000
    if len(input_content) > max_input_length:
        # Take beginning and end (often has key info)
        input_content = input_content[:max_input_length//2] + "\n...[truncated]...\n" + input_content[-max_input_length//2:]

    prompt = f'''I have a template with highlighted placeholders that need to be filled with data extracted from input documents.

## TEMPLATE PLACEHOLDERS TO FILL:
{placeholders_list}

## INPUT DOCUMENTS DATA:
{input_content}

## YOUR TASK:
1. Read the input documents carefully
2. For EACH placeholder above, find the corresponding value from the input data
3. Return a JSON object mapping each placeholder to its extracted value

## RULES:
- Use EXACT text from input documents where possible
- For names: Extract the actual client/person name from the input
- For dates: Use the format found in the input
- For amounts: Include currency symbols (£, $) if present
- If a value cannot be found, use "N/A"
- DO NOT make up data - only use what's in the input

## RESPONSE FORMAT (JSON only):
{{
  "placeholder1": "extracted value 1",
  "placeholder2": "extracted value 2",
  ...
}}

Return ONLY the JSON object, no other text:'''

    if progress_callback:
        progress_callback("Calling Azure OpenAI for intelligent mapping...")

    response = call_azure_openai(prompt, timeout=60)

    # Parse JSON response
    try:
        # Find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            mappings = json.loads(json_match.group(0))
            return mappings
    except json.JSONDecodeError as e:
        st.warning(f"Could not parse LLM response as JSON: {e}")

    return {}


def generate_filled_document(template_bytes: bytes, mappings: Dict[str, str]) -> bytes:
    """Generate filled Word document by replacing highlighted text"""
    template_io = io.BytesIO(template_bytes)
    output_io = io.BytesIO()

    with zipfile.ZipFile(template_io, 'r') as zin:
        with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                data = zin.read(item)

                if item == 'word/document.xml':
                    content = data.decode('utf-8')

                    # Apply each mapping
                    for original, replacement in mappings.items():
                        if replacement and replacement != 'N/A' and replacement != original:
                            # Escape XML special characters
                            safe_replacement = (str(replacement)
                                .replace('&', '&amp;')
                                .replace('<', '&lt;')
                                .replace('>', '&gt;')
                                .replace('"', '&quot;')
                                .replace("'", '&apos;'))

                            # Replace in content
                            content = content.replace(original, safe_replacement)

                    data = content.encode('utf-8')

                zout.writestr(item, data)

    return output_io.getvalue()


def main():
    st.title("🧠 Intelligent Template Builder")
    st.markdown("**Upload input files → Upload template → AI extracts & fills automatically**")

    # Initialize session state
    if 'processed_doc' not in st.session_state:
        st.session_state.processed_doc = None
    if 'mappings' not in st.session_state:
        st.session_state.mappings = {}
    if 'highlights' not in st.session_state:
        st.session_state.highlights = []

    # API Status
    with st.expander("☁️ Azure OpenAI Status", expanded=False):
        st.success(f"**Connected** | Endpoint: {AZURE_ENDPOINT} | Deployment: {AZURE_DEPLOYMENT}")

    # File upload columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Step 1: Upload Input Files")
        st.caption("Upload client data files (Word, PDF, TXT) containing information to extract")
        input_files = st.file_uploader(
            "Select input files",
            type=['docx', 'pdf', 'txt'],
            accept_multiple_files=True,
            key="inputs",
            label_visibility="collapsed"
        )
        if input_files:
            st.success(f"✅ {len(input_files)} file(s) uploaded")
            for f in input_files:
                st.caption(f"  • {f.name}")

    with col2:
        st.subheader("📋 Step 2: Upload Template")
        st.caption("Upload Word template with highlighted sections to fill")
        template_file = st.file_uploader(
            "Select template",
            type=['docx'],
            key="template",
            label_visibility="collapsed"
        )
        if template_file:
            st.success(f"✅ Template: {template_file.name}")

    st.divider()

    # Process button
    can_process = input_files and template_file

    if st.button("🧠 Extract Data & Fill Template", type="primary", disabled=not can_process, use_container_width=True):

        # Reset state
        st.session_state.processed_doc = None
        st.session_state.mappings = {}

        # Progress container
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()

        def log(message):
            with log_container:
                st.text(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

        try:
            # STEP 1: Read all input files
            status_text.text("📁 Reading input files...")
            progress_bar.progress(10)

            combined_input = ""
            for i, file in enumerate(input_files):
                log(f"Reading: {file.name}")

                if file.name.endswith('.docx'):
                    text, _, _ = read_word_document(file)
                elif file.name.endswith('.pdf'):
                    text = read_pdf_document(file)
                else:
                    text = read_text_file(file)

                combined_input += f"\n\n=== {file.name} ===\n{text}"
                progress_bar.progress(10 + int(20 * (i+1) / len(input_files)))

            log(f"Total input: {len(combined_input)} characters")

            # STEP 2: Read template and detect highlights
            status_text.text("📋 Analyzing template for highlighted sections...")
            progress_bar.progress(35)

            _, template_xml, template_bytes = read_word_document(template_file)
            highlights = detect_all_highlights(template_xml)
            st.session_state.highlights = highlights

            log(f"Found {len(highlights)} highlighted sections to fill")

            if not highlights:
                st.warning("⚠️ No highlighted sections found in template. Make sure text is highlighted (yellow, blue, etc.)")
                return

            # Show what was found
            with st.expander(f"📝 Found {len(highlights)} placeholders to fill", expanded=True):
                for h in highlights[:20]:
                    st.text(f"  • \"{h['text']}\" ({h['color']})")
                if len(highlights) > 20:
                    st.text(f"  ... and {len(highlights) - 20} more")

            # STEP 3: Use LLM to extract and map data
            status_text.text("🧠 AI is extracting data and mapping to template...")
            progress_bar.progress(50)
            log("Calling Azure OpenAI for intelligent data extraction...")

            start_time = time.time()

            try:
                mappings = extract_and_map_with_llm(
                    combined_input,
                    highlights,
                    progress_callback=log
                )

                elapsed = time.time() - start_time
                log(f"LLM completed in {elapsed:.1f}s - mapped {len(mappings)} fields")

            except Exception as e:
                log(f"LLM Error: {e}")
                st.error(f"AI extraction failed: {e}")
                return

            st.session_state.mappings = mappings
            progress_bar.progress(80)

            # STEP 4: Generate filled document
            status_text.text("📝 Generating filled document...")
            log("Applying mappings to template...")

            filled_doc = generate_filled_document(template_bytes, mappings)
            st.session_state.processed_doc = filled_doc

            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            log("Document generated successfully!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            log(f"Error: {str(e)}")

    # Results section
    if st.session_state.processed_doc:
        st.divider()
        st.subheader("✅ Document Ready!")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Generate filename
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Filled_Template_{date_str}.docx"

            st.download_button(
                label="📥 Download Filled Document",
                data=st.session_state.processed_doc,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True
            )

        with col2:
            total = len(st.session_state.highlights)
            filled = sum(1 for v in st.session_state.mappings.values() if v and v != 'N/A')
            st.metric("Sections Filled", f"{filled}/{total}", f"{int(filled/total*100) if total > 0 else 0}%")

        # Show mappings
        if st.session_state.mappings:
            with st.expander("🔄 View All Mappings", expanded=False):
                for original, value in st.session_state.mappings.items():
                    if value and value != 'N/A':
                        st.text(f"✅ \"{original[:40]}...\" → \"{value[:60]}{'...' if len(str(value)) > 60 else ''}\"")
                    else:
                        st.text(f"❌ \"{original[:40]}...\" → Not found")


if __name__ == "__main__":
    main()
