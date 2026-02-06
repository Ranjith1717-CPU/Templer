# Intelligent Template Builder

An AI-powered document processing application that automatically extracts data from client reports and fills professional templates using Azure OpenAI.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Chosen Problem

**Document Automation for Financial Advisory Services**

Financial advisors spend hours manually extracting client information from multiple source documents (fact finds, risk assessments, meeting notes) and copying this data into standardized report templates. This manual process is:

- Time-consuming (2-4 hours per client report)
- Error-prone (copy-paste mistakes, missed fields)
- Inconsistent (different advisors fill templates differently)
- Not scalable (limits advisor capacity)

---

## Solution Overview

**Intelligent Template Builder** solves this by using AI to:

1. **Upload Multiple Client Reports** - Accept Word (.docx), PDF, and text files containing client information
2. **Upload Output Template** - Accept a Word template with highlighted sections that need to be filled
3. **AI-Powered Extraction** - Use Azure OpenAI to intelligently extract relevant data from all uploaded documents
4. **Smart Mapping** - Automatically match extracted data to template placeholders
5. **Generate Filled Document** - Produce a professionally filled Word document ready for client delivery

### Key Features

- Multi-file upload support (Word, PDF, TXT)
- Automatic client name detection
- Highlighted section detection in Word templates
- Azure OpenAI integration for intelligent data extraction
- Pattern-based fallback extraction (works even without API)
- Preserves original template formatting
- One-click download of filled documents

---

## Tech Stack Used

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.9+ |
| **AI/LLM** | Azure OpenAI (GPT-4) |
| **Document Processing** | python-docx, pdfplumber |
| **Data Handling** | pandas |
| **HTTP Requests** | requests |
| **Hosting** | Streamlit Cloud |

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Client Reports │────>│   Streamlit     │────>│  Azure OpenAI   │
│  (Word/PDF/TXT) │     │   Application   │     │  (Data Extract) │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
┌─────────────────┐              │              ┌─────────────────┐
│  Word Template  │──────────────┤              │  Filled Word    │
│  (Highlighted)  │              └─────────────>│  Document       │
└─────────────────┘                             └─────────────────┘
```

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Azure OpenAI API access (or use the pre-configured credentials)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/intelligent-template-builder.git
   cd intelligent-template-builder
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Environment Variables

The application uses Azure OpenAI for AI-powered data extraction. Configure these environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_API_KEY` | Your Azure OpenAI API key | Yes |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint URL | Yes |
| `AZURE_DEPLOYMENT` | Model deployment name | Yes |
| `AZURE_API_VERSION` | API version (e.g., 2024-12-01-preview) | Yes |

### Setting up Secrets

**For Local Development:**

Create a file `.streamlit/secrets.toml`:

```toml
AZURE_API_KEY = "your-azure-openai-api-key"
AZURE_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_DEPLOYMENT = "your-deployment-name"
AZURE_API_VERSION = "2024-12-01-preview"
```

**For Streamlit Cloud:**

Add secrets in your app's Settings > Secrets section with the same format.

---

## Step-by-Step Guide to Run the Project Locally

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/intelligent-template-builder.git
cd intelligent-template-builder

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Azure OpenAI Credentials

```bash
# Create secrets directory
mkdir -p .streamlit

# Create secrets file
# Windows (PowerShell):
New-Item -Path ".streamlit/secrets.toml" -ItemType File

# macOS/Linux:
touch .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your credentials:

```toml
AZURE_API_KEY = "your-api-key-here"
AZURE_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_DEPLOYMENT = "your-deployment-name"
AZURE_API_VERSION = "2024-12-01-preview"
```

### Step 3: Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Step 4: Use the Application

1. **Upload Client Reports**
   - Click "Browse files" in the left panel
   - Select one or more client documents (Word, PDF, or TXT)
   - Supported: Fact finds, risk assessments, meeting notes, etc.

2. **Upload Template**
   - Click "Browse files" in the right panel
   - Select your Word template (.docx) with highlighted sections

3. **Process**
   - Click "Analyze Reports & Fill Template with Azure OpenAI"
   - Wait for processing to complete

4. **Download**
   - Click "Download Filled Document"
   - Your filled template is ready!

---

## Demo

### Live Application
[Link to your deployed Streamlit app]

### Sample Files
Sample client reports and templates are available in the `test-inputs/` folder for testing.

---

## Project Structure

```
intelligent-template-builder/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .streamlit/
    └── secrets.toml.example  # Environment variables template
```

---

## License

MIT License

---

## Author

Built for the Hackathon 2025

---

## Acknowledgments

- Azure OpenAI for powering the intelligent data extraction
- Streamlit for the amazing web framework
- python-docx for Word document processing
