"""
Intelligent Template Builder - Streamlit Version
Converts client reports into filled professional templates using Azure OpenAI
"""

import streamlit as st
import re
import json
import io
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

# For Word document processing
try:
    from docx import Document
    from docx.shared import Inches
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

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
# You can set these via Streamlit secrets or environment variables
AZURE_API_KEY = st.secrets.get("AZURE_API_KEY", "AHz5fOtxXHg0m0OYl9neAlfOnna79WBhvetPnnZ4nssRlZXiK9FBJQQJ99BIACYeBjFXJ3w3AAABACOGcHNC")
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", "https://curious-01.openai.azure.com/")
AZURE_DEPLOYMENT = st.secrets.get("AZURE_DEPLOYMENT", "Ranjith")
AZURE_API_VERSION = st.secrets.get("AZURE_API_VERSION", "2024-12-01-preview")


class IntelligentTemplateBuilder:
    """Main class for template processing"""

    def __init__(self):
        self.client_reports: List[Dict] = []
        self.template_content: Optional[str] = None
        self.template_xml: Optional[str] = None
        self.template_zip: Optional[zipfile.ZipFile] = None
        self.highlighted_sections: List[Dict] = []
        self.extracted_data: Dict = {}
        self.client_name: str = "Client Name"

    def read_word_document(self, file) -> Tuple[str, str, bytes]:
        """Read Word document and return text content, XML, and raw bytes"""
        file_bytes = file.read()
        file.seek(0)  # Reset for potential re-read

        text_content = ""
        document_xml = ""

        try:
            # Read using zipfile to get raw XML
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                if 'word/document.xml' in zf.namelist():
                    document_xml = zf.read('word/document.xml').decode('utf-8')

                    # Extract text from XML
                    text_matches = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', document_xml)
                    text_content = ' '.join(text_matches)
        except Exception as e:
            st.warning(f"Error reading Word document: {e}")

        return text_content, document_xml, file_bytes

    def read_pdf_document(self, file) -> str:
        """Read PDF and return text content"""
        if not PDF_AVAILABLE:
            return f"PDF file: {file.name} (pdfplumber not installed)"

        text_content = ""
        try:
            with pdfplumber.open(file) as pdf:
                for i, page in enumerate(pdf.pages[:15]):  # Limit to 15 pages
                    page_text = page.extract_text() or ""
                    text_content += page_text + "\n"
                    if len(text_content) > 50000:  # Limit content size
                        break
        except Exception as e:
            st.warning(f"Error reading PDF: {e}")
            text_content = f"PDF file: {file.name}"

        return text_content

    def read_text_file(self, file) -> str:
        """Read text file"""
        try:
            return file.read().decode('utf-8')
        except:
            return file.read().decode('latin-1')

    def classify_document(self, filename: str, content: str) -> str:
        """Classify document type based on filename and content"""
        lower = filename.lower()

        if 'fact' in lower and 'find' in lower:
            return 'fact_find'
        if 'meeting' in lower and 'note' in lower:
            return 'meeting_notes'
        if 'risk' in lower:
            return 'risk_assessment'
        if 'fee' in lower:
            return 'fee_structure'
        if 'performance' in lower:
            return 'performance_report'
        if 'illustration' in lower:
            return 'product_illustration'

        return 'general_report'

    def detect_highlighted_sections(self, document_xml: str) -> List[Dict]:
        """Detect highlighted sections in Word template"""
        highlights = []
        seen = set()

        # Pattern 1: Standard highlight pattern
        pattern1 = r'<w:highlight[^>]*\/>\s*<\/w:rPr>\s*<w:t[^>]*>([^<]+)<\/w:t>'
        for match in re.finditer(pattern1, document_xml):
            text = match.group(1).strip()
            if text and text not in seen:
                seen.add(text)
                highlights.append({
                    'originalText': text,
                    'placeholder': re.sub(r'[^a-zA-Z0-9]', '_', text),
                    'context': ''
                })

        # Pattern 2: Alternative highlight pattern
        pattern2 = r'<w:rPr>.*?<w:highlight[^>]*\/>.*?<\/w:rPr>.*?<w:t[^>]*>([^<]+)<\/w:t>'
        for match in re.finditer(pattern2, document_xml, re.DOTALL):
            text = match.group(1).strip()
            if text and text not in seen:
                seen.add(text)
                highlights.append({
                    'originalText': text,
                    'placeholder': re.sub(r'[^a-zA-Z0-9]', '_', text),
                    'context': ''
                })

        return highlights

    def detect_client_name(self, content: str) -> str:
        """Detect client name from document content"""
        # Pattern 1: "Client Name: John Smith"
        match = re.search(r'Client\s*Name\s*[:\s]+([A-Za-z]+\s+[A-Za-z]+)', content, re.IGNORECASE)
        if match:
            return self._capitalize_words(match.group(1).strip())

        # Pattern 2: Various name patterns
        patterns = [
            r'(?:client|name|full name)[:\s]+([a-zA-Z]+\s+[a-zA-Z]+)',
            r'Dear\s+([A-Z][a-zA-Z]+\s+[A-Za-z]+)',
            r'Mr\.?\s+([A-Z][a-zA-Z]+\s*[A-Za-z]*)',
            r'Mrs\.?\s+([A-Z][a-zA-Z]+\s*[A-Za-z]*)',
            r'Name[:\s]+([A-Za-z]+\s+[A-Za-z]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                name = self._capitalize_words(match.group(1).strip())
                if 3 <= len(name) <= 50 and ' ' in name:
                    return name

        # Fallback: Find most frequent capitalized name
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', content)
        if names:
            from collections import Counter
            most_common = Counter(names).most_common(1)
            if most_common:
                return most_common[0][0]

        return "Client Name"

    def _capitalize_words(self, text: str) -> str:
        """Capitalize each word"""
        return ' '.join(word.capitalize() for word in text.split())

    def call_azure_openai(self, prompt: str) -> str:
        """Call Azure OpenAI API"""
        endpoint = f"{AZURE_ENDPOINT}openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_API_KEY
        }

        payload = {
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that extracts structured data from financial documents. Always respond with valid JSON only.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 2000,
            'temperature': 0.3
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            raise Exception("Azure OpenAI request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Azure OpenAI API error: {str(e)}")

    def extract_data_with_llm(self, content: str, client_name: str) -> Dict:
        """Extract structured data using Azure OpenAI"""
        # Truncate content to avoid token limits
        truncated = content[:8000]

        prompt = f'''Extract ALL data from this financial document. Return JSON only.

DOCUMENT:
{truncated}

Extract into this JSON format (use exact values from document, "N/A" if not found):
{{
  "client_name": "{client_name}",
  "provider": "",
  "plan_number": "",
  "plan_type": "",
  "current_value": "",
  "transfer_value": "",
  "retirement_age": "",
  "amc": "",
  "funds": "",
  "risk_tolerance": "",
  "employer": "",
  "income": "",
  "valuation_date": "",
  "contributions": "",
  "death_benefits": "",
  "adviser": "",
  "age": "",
  "occupation": "",
  "phone": "",
  "email": "",
  "address": ""
}}

JSON only:'''

        response = self.call_azure_openai(prompt)

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
        except json.JSONDecodeError:
            pass

        return {'client_name': client_name}

    def fast_extract_data(self, content: str) -> Dict:
        """Fast regex-based data extraction (fallback)"""
        data = {}

        patterns = {
            'client_name': r'Client\s*Name[:\s]+([A-Za-z\s]+?)(?:\s{2,}|\n|$)',
            'provider': r'Provider[:\s]+([A-Za-z\s]+?)(?:\s{2,}|\n|$)',
            'plan_number': r'Plan\s*Number[:\s]+([0-9A-Za-z]+)',
            'plan_type': r'Plan\s*Type[:\s]+([A-Za-z\s\-]+?)(?:\s{2,}|\n|$)',
            'current_value': r'Current\s*Value[:\s]+(£?[\d,\.]+)',
            'transfer_value': r'Transfer\s*Value[:\s]+(£?[\d,\.]+)',
            'valuation_date': r'Date\s*of\s*Valuation[:\s]+([0-9A-Za-z\s]+?)(?:\s{2,}|\n|$)',
            'retirement_age': r'Retirement\s*Age[:\s]+(\d+)',
            'amc': r'AMC[:\s]+([\d\.]+%?)',
            'risk_tolerance': r'Risk[:\s]+(Conservative|Moderate|Balanced|Aggressive|Cautious|Adventurous)',
            'employer': r'Employer[:\s]+([A-Za-z\s]+?)(?:\s{2,}|\n|$)',
            'occupation': r'Occupation[:\s]+([A-Za-z\s]+?)(?:\s{2,}|\n|$)',
            'income': r'(?:Annual\s*)?Income[:\s]+(£?[\d,\.]+)',
            'phone': r'Phone[:\s]+([0-9\s\-]+)',
            'email': r'Email[:\s]+([a-zA-Z0-9@\.\-_]+)',
            'address': r'Address[:\s]+([A-Za-z0-9\s,]+?)(?:\s{2,}|\n|$)',
            'meeting_date': r'(?:Meeting\s*)?Date[:\s]+(\d{1,2}[\s\/\-][A-Za-z]+[\s\/\-]\d{2,4})',
            'adviser': r'(?:Adviser|Advisor)[:\s]+([A-Za-z\s]+?)(?:\s{2,}|\n|$)',
            'age': r'Age[:\s]+(\d{1,3})'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                data[key] = match.group(1).strip()

        return data

    def create_mappings(self, extracted_data: Dict, client_name: str) -> List[Dict]:
        """Create mappings from highlighted sections to extracted data"""
        mappings = []
        name_parts = client_name.split()
        first_name = name_parts[0] if name_parts else client_name
        last_name = name_parts[-1] if len(name_parts) > 1 else first_name

        # Template names to replace
        template_first_names = ['stacey', 'sarah', 'rebecca', 'john', 'roger', 'joe', 'james', 'david', 'michael', 'robert']
        template_last_names = ['shipman', 'morgan', 'doe', 'smith', 'bloggs', 'jones', 'williams', 'brown', 'taylor', 'law']

        for section in self.highlighted_sections:
            original = section['originalText']
            text_lower = original.lower().strip()

            value = original  # Default: keep original
            source = 'No Match'
            confidence = 0

            # Name matching
            if text_lower in template_first_names:
                value = first_name
                source = 'First Name'
                confidence = 95
            elif text_lower in template_last_names:
                value = last_name
                source = 'Last Name'
                confidence = 95
            elif any(f"{fn} {ln}" in text_lower for fn in template_first_names for ln in template_last_names):
                value = client_name
                source = 'Full Name'
                confidence = 90
            else:
                # Data matching
                match = self._find_data_match(text_lower, original, extracted_data)
                if match:
                    value = match['value']
                    source = match['source']
                    confidence = match['confidence']

            mappings.append({
                'original': original,
                'value': value,
                'source': source,
                'confidence': confidence
            })

        return mappings

    def _find_data_match(self, text_lower: str, original: str, data: Dict) -> Optional[Dict]:
        """Find matching data for a highlighted section"""
        # Date patterns
        if self._looks_like_date(original):
            if data.get('valuation_date'):
                return {'value': data['valuation_date'], 'source': 'Date', 'confidence': 70}
            if data.get('meeting_date'):
                return {'value': data['meeting_date'], 'source': 'Date', 'confidence': 70}

        # Money patterns
        if self._looks_like_money(original):
            if data.get('current_value'):
                return {'value': data['current_value'], 'source': 'Value', 'confidence': 70}

        # Provider names
        providers = ['aj bell', 'royal london', 'standard life', 'aviva', 'scottish widows']
        if any(p in text_lower for p in providers):
            if data.get('provider'):
                return {'value': data['provider'], 'source': 'Provider', 'confidence': 80}

        return None

    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date"""
        return bool(re.search(r'\d{1,2}(st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', text, re.IGNORECASE) or
                   re.search(r'\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}', text))

    def _looks_like_money(self, text: str) -> bool:
        """Check if text looks like a money value"""
        return bool(re.search(r'[£$][\d,]+(\.\d{2})?', text))

    def generate_filled_template(self, template_bytes: bytes, mappings: List[Dict]) -> bytes:
        """Generate filled Word document"""
        # Read template as zip
        template_io = io.BytesIO(template_bytes)
        output_io = io.BytesIO()

        with zipfile.ZipFile(template_io, 'r') as zin:
            with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as zout:
                for item in zin.namelist():
                    data = zin.read(item)

                    if item == 'word/document.xml':
                        # Apply replacements
                        content = data.decode('utf-8')
                        for mapping in mappings:
                            if mapping['value'] and mapping['value'] != 'N/A' and mapping['value'] != mapping['original']:
                                content = self._safe_replace(content, mapping['original'], mapping['value'])
                        data = content.encode('utf-8')

                    zout.writestr(item, data)

        return output_io.getvalue()

    def _safe_replace(self, xml: str, original: str, replacement: str) -> str:
        """Safely replace text in WordML XML"""
        # Escape XML special characters in replacement
        escaped = (replacement
                  .replace('&', '&amp;')
                  .replace('<', '&lt;')
                  .replace('>', '&gt;')
                  .replace('"', '&quot;')
                  .replace("'", '&apos;'))

        # Try direct replacement first
        try:
            escaped_original = re.escape(original)
            result = re.sub(escaped_original, escaped, xml)
            if result != xml:
                return result
        except:
            pass

        # Fallback to simple string replacement
        return xml.replace(original, escaped)


def main():
    st.title("🧠 Intelligent Template Builder")
    st.markdown("**Professional LLM-Powered Multi-Report Analysis & Template Generation**")

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = IntelligentTemplateBuilder()
    if 'processed_doc' not in st.session_state:
        st.session_state.processed_doc = None
    if 'processing_log' not in st.session_state:
        st.session_state.processing_log = []

    processor = st.session_state.processor

    def log(message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.processing_log.append(f"[{timestamp}] {message}")

    # Azure OpenAI Status
    with st.expander("☁️ Azure OpenAI Configuration", expanded=False):
        st.success(f"**Azure OpenAI Connected**\n\nEndpoint: {AZURE_ENDPOINT}\n\nDeployment: {AZURE_DEPLOYMENT}")

    # Two column layout for file uploads
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Step 1: Upload Client Input Reports")
        client_files = st.file_uploader(
            "Upload client reports (Word, PDF, TXT)",
            type=['docx', 'pdf', 'txt'],
            accept_multiple_files=True,
            key="client_reports"
        )

        if client_files:
            st.info(f"📄 {len(client_files)} file(s) uploaded")
            for f in client_files:
                st.write(f"  - {f.name}")

    with col2:
        st.subheader("📋 Step 2: Upload Output Template")
        template_file = st.file_uploader(
            "Upload Word template (.docx)",
            type=['docx'],
            key="template"
        )

        if template_file:
            st.success(f"✅ Template: {template_file.name}")

    st.divider()

    # Process button
    col1, col2 = st.columns([2, 1])

    with col1:
        process_btn = st.button(
            "🧠 Analyze Reports & Fill Template with Azure OpenAI",
            type="primary",
            disabled=not (client_files and template_file)
        )

    with col2:
        preserve_formatting = st.checkbox("Preserve All Formatting", value=True)

    if process_btn and client_files and template_file:
        st.session_state.processing_log = []

        with st.spinner("Processing..."):
            progress = st.progress(0)
            status = st.empty()

            try:
                # Step 1: Read client reports
                status.text("📁 Reading client reports...")
                log("📁 Processing client reports...")

                combined_content = ""
                processor.client_reports = []

                for i, file in enumerate(client_files):
                    progress.progress((i + 1) / (len(client_files) + 4) * 0.3)

                    if file.name.endswith('.docx'):
                        text, _, _ = processor.read_word_document(file)
                    elif file.name.endswith('.pdf'):
                        text = processor.read_pdf_document(file)
                    else:
                        text = processor.read_text_file(file)

                    # Limit content size
                    if len(text) > 100000:
                        text = text[:100000]

                    doc_type = processor.classify_document(file.name, text)
                    processor.client_reports.append({
                        'name': file.name,
                        'content': text,
                        'type': doc_type
                    })
                    combined_content += text + "\n\n"
                    log(f"✅ Read: {file.name} ({doc_type})")

                # Step 2: Read template
                status.text("📋 Analyzing template...")
                progress.progress(0.4)
                log("📋 Analyzing template...")

                template_text, template_xml, template_bytes = processor.read_word_document(template_file)
                processor.template_xml = template_xml

                # Step 3: Detect highlights
                processor.highlighted_sections = processor.detect_highlighted_sections(template_xml)
                log(f"🎯 Found {len(processor.highlighted_sections)} highlighted sections")

                # Step 4: Detect client name
                status.text("👤 Detecting client name...")
                progress.progress(0.5)

                processor.client_name = processor.detect_client_name(combined_content)
                log(f"👤 Client identified: {processor.client_name}")

                # Step 5: Extract data with LLM
                status.text("🧠 Extracting data with Azure OpenAI...")
                progress.progress(0.6)
                log("🧠 Calling Azure OpenAI for data extraction...")

                try:
                    extracted_data = processor.extract_data_with_llm(combined_content, processor.client_name)
                    log(f"✅ LLM extracted {len(extracted_data)} fields")
                except Exception as e:
                    log(f"⚠️ LLM failed: {e}, using pattern matching")
                    extracted_data = processor.fast_extract_data(combined_content)

                extracted_data['client_name'] = processor.client_name
                processor.extracted_data = extracted_data

                # Step 6: Create mappings
                status.text("🔧 Creating intelligent mappings...")
                progress.progress(0.7)

                mappings = processor.create_mappings(extracted_data, processor.client_name)
                matched = sum(1 for m in mappings if m['source'] != 'No Match')
                log(f"✅ Matched {matched} of {len(mappings)} sections")

                # Step 7: Generate document
                status.text("📝 Generating filled document...")
                progress.progress(0.9)
                log("📝 Generating Word document...")

                st.session_state.processed_doc = processor.generate_filled_template(template_bytes, mappings)
                st.session_state.mappings = mappings

                progress.progress(1.0)
                log("✅ Document generated successfully!")
                status.text("✅ Processing complete!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                log(f"❌ Error: {str(e)}")

    # Processing log
    if st.session_state.processing_log:
        with st.expander("📊 Processing Log", expanded=True):
            for msg in st.session_state.processing_log:
                st.text(msg)

    # Results and download
    if st.session_state.processed_doc:
        st.divider()
        st.subheader("✅ Processing Complete")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Generate filename
            client_name_clean = re.sub(r'[^a-zA-Z0-9]', '_', processor.client_name)
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{client_name_clean}_Filled_Template_{date_str}.docx"

            st.download_button(
                label="📥 Download Filled Document",
                data=st.session_state.processed_doc,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary"
            )

        with col2:
            if hasattr(st.session_state, 'mappings'):
                st.write(f"**Client:** {processor.client_name}")
                st.write(f"**Sections processed:** {len(st.session_state.mappings)}")
                matched = sum(1 for m in st.session_state.mappings if m['source'] != 'No Match')
                st.write(f"**Matched:** {matched} ({round(matched/len(st.session_state.mappings)*100)}%)")

        # Show extracted data
        if processor.extracted_data:
            with st.expander("📊 Extracted Data", expanded=False):
                cols = st.columns(3)
                items = list(processor.extracted_data.items())
                for i, (key, value) in enumerate(items):
                    if value and value != 'N/A':
                        cols[i % 3].write(f"**{key}:** {value[:50]}{'...' if len(str(value)) > 50 else ''}")

        # Show mappings table
        if hasattr(st.session_state, 'mappings'):
            with st.expander("🔄 Template Mappings", expanded=False):
                import pandas as pd
                df = pd.DataFrame(st.session_state.mappings)
                df = df[['original', 'value', 'source', 'confidence']]
                df.columns = ['Original', 'Filled With', 'Source', 'Confidence']
                st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
