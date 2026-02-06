# Templer v2.0

**AI-Powered Template Intelligence Engine for UK Financial Advisors**

Zero-prep template processing with auto-detection, learning, and validation.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://templer.streamlit.app/)

---

## The Problem We Solve

### The Template Setup Nightmare

When a new advisory firm signs up with five different report templates—pensions, investments, protection, annual reviews, ad-hoc advice—each template is a Word document with carefully crafted formatting that took their compliance team months to approve.

**The Old Process:**
- Delivery team manually identifies static vs. dynamic sections
- Insert placeholders, write prompts, test outputs
- Formatting breaks → fix formatting → tone is wrong → adjust prompts
- Tables don't render → rebuild tables
- **Each template: ~4 hours**
- **5 templates × 4 hours = 20+ hours per firm before generating a single report**

**This doesn't scale.**

### The Templer v2.0 Solution

| Before | After (Templer v2.0) |
|--------|---------------------|
| 4 hours per template | 5-10 minutes |
| Manual placeholder insertion | Auto-detection |
| Engineers inspect XML | Zero-prep upload |
| Formatting constantly breaks | Format preserved |
| 20+ hours per firm | < 1 hour per firm |

---

## What's New in v2.0

### Four Operation Modes

| Mode | Description | Best For |
|------|-------------|----------|
| 🔧 **Analyze & Create Template** | **NEW!** Upload raw template → AI identifies static vs dynamic → Creates template with placeholders | **Solving the 4-hour setup problem** |
| 🔍 **Auto-Detect & Fill** | Automatically finds insertion points using patterns & AI | Filling templates with data |
| 📚 **Learn from Example** | Upload blank + filled pair, system learns what changes | Compliance-approved templates |
| 🖍️ **Manual Highlights** | Original behavior - use highlighted text | Legacy templates |

### Key Feature: Template Analyzer (NEW!)

The **Analyze & Create Template** mode solves the core problem:

```
RAW TEMPLATE (firm's Word doc)           OUTPUT TEMPLATE (ready for use)
┌──────────────────────────────┐         ┌──────────────────────────────┐
│ Dear John Smith,             │    →    │ Dear {{client_name}},        │
│ We recommend transferring    │    →    │ {{LLM:recommendation}}       │
│ £150,000 from your pension   │    →    │ {{pension_value}} from your  │
│ Risk Warning: Past...        │    →    │ Risk Warning: Past...        │
└──────────────────────────────┘         └──────────────────────────────┘
        4 hours manual                           5 minutes review
```

**What it does:**
1. **Classifies content**: Static (legal, boilerplate) vs Dynamic (client data, recommendations)
2. **Inserts placeholders**: `{{client_name}}`, `{{LLM:recommendation}}`, etc.
3. **Generates prompt hints**: For each LLM-generated section
4. **Creates downloadable template**: Ready for production use

### Other Key Capabilities

- **Zero Template Preparation**: Upload any Word template, system detects fields automatically
- **Pattern Recognition**: Finds `{{placeholders}}`, dates, currencies, names, percentages
- **Context Analysis**: Uses headings like "Client Name:" to understand field purpose
- **Learning Engine**: Show one example, system learns the pattern
- **Format Preservation**: Handles split runs, tables, merged cells
- **Confidence Scoring**: Every field shows 0-100% detection confidence
- **Conditional Logic**: Support for `[[IF condition]]...[[/IF]]` sections

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEMPLER v2.0 ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MODE 1: ANALYZE & CREATE TEMPLATE (Solves 4-hour problem)      │
│  ─────────────────────────────────────────────────────────────  │
│  Raw Template ──► Template Analyzer ──► Static/Dynamic Classify │
│                         │                        │              │
│                         ▼                        ▼              │
│              Placeholder Insertion    Prompt Config Generation  │
│                         │                        │              │
│                         ▼                        ▼              │
│              Output Template.docx     prompt_config.json        │
│                                                                 │
│  MODE 2-4: FILL TEMPLATE (Using prepared template)              │
│  ─────────────────────────────────────────────────────────────  │
│  Template Input ──┬──► Auto-Detection Engine                    │
│                   ├──► Learning Engine (blank + filled pair)    │
│                   └──► Manual Highlights (fallback)             │
│                              │                                  │
│                              ▼                                  │
│                   Unified Insertion Point Registry              │
│                              │                                  │
│                              ▼                                  │
│  Input Data ────► Azure OpenAI ────► Smart Content Mapper       │
│                              │                                  │
│                              ▼                                  │
│                   Format-Preserving Replacer                    │
│                   (handles split runs, tables, conditionals)    │
│                              │                                  │
│                              ▼                                  │
│                   Validation & Confidence Engine                │
│                              │                                  │
│                              ▼                                  │
│                        Output .docx                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Mode 1: Auto-Detect (Recommended)

```
Upload Template → System scans for patterns → Detects 80%+ of fields → Ready to fill
```

**What it detects automatically:**
- Explicit placeholders: `{{client_name}}`, `[CLIENT NAME]`, `${amount}`
- Currency values: `£150,000`, `$50,000`
- Percentages: `5.5%`, `+12.3%`
- Dates: `15/03/2024`, `March 15, 2024`
- Names with context: `Client Name: John Smith`
- References: `Policy No: ABC123`

### Mode 2: Learn from Example

```
Upload Blank Template + Filled Example → System diffs them → Learns what changed → Creates schema
```

**Perfect for:**
- Compliance-approved templates you can't modify
- Templates with no explicit placeholders
- Complex documents where context matters

### Mode 3: Manual Highlights (Original)

```
Highlight text in Word → Upload → Highlighted sections become placeholders
```

**Use when:**
- You have existing highlighted templates
- You want precise control over what gets replaced

---

## Quick Start

### Option 1: Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your fork
4. Add Azure OpenAI secrets in Settings

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/intelligent-template-builder.git
cd intelligent-template-builder/streamlit-deploy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure Azure OpenAI (create .streamlit/secrets.toml)
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
AZURE_API_KEY = "your-api-key"
AZURE_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_DEPLOYMENT = "your-deployment-name"
AZURE_API_VERSION = "2024-12-01-preview"
EOF

# Run the app
streamlit run templer.py
```

Opens at `http://localhost:8501`

---

## Project Structure

```
streamlit-deploy/
├── templer.py                 # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
└── templer_core/              # Core intelligence modules
    ├── __init__.py            # Package exports
    ├── patterns.py            # Regex patterns for detection
    ├── auto_detector.py       # Auto-detection engine
    ├── learning_engine.py     # Learn from examples
    ├── document_parser.py     # Deep OOXML parsing
    ├── table_processor.py     # Table handling
    ├── format_preserver.py    # Split run & formatting
    ├── conditionals.py        # IF/THEN logic
    └── validation.py          # Confidence scoring
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `patterns.py` | 20+ regex patterns for dates, currencies, names, placeholders |
| `auto_detector.py` | 5-method detection: patterns, highlights, dynamic values, context, heuristics |
| `learning_engine.py` | Diff-based learning from blank/filled template pairs |
| `document_parser.py` | Deep OOXML parsing, heading hierarchy, structure extraction |
| `table_processor.py` | Table analysis, row cloning, merged cell handling |
| `format_preserver.py` | Split run handling, formatting preservation |
| `conditionals.py` | `[[IF]]`, `{% if %}`, `{{#if}}` conditional blocks |
| `validation.py` | Confidence scoring, XML validation, output quality checks |

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint URL | Yes |
| `AZURE_DEPLOYMENT` | Model deployment name | Yes |
| `AZURE_API_VERSION` | API version (e.g., `2024-12-01-preview`) | Yes |

### For Streamlit Cloud

Add secrets in **Settings → Secrets**:

```toml
AZURE_API_KEY = "your-api-key"
AZURE_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_DEPLOYMENT = "your-deployment-name"
AZURE_API_VERSION = "2024-12-01-preview"
```

---

## Usage Guide

### Step 1: Choose Detection Mode

In the sidebar, select:
- **Auto-Detect** for new templates (no prep needed)
- **Learn from Example** if you have a filled example
- **Manual Highlights** for highlighted templates

### Step 2: Upload Files

**For Auto-Detect / Manual Highlights:**
1. Upload input data files (fact finds, meeting notes, etc.)
2. Upload your Word template

**For Learning Mode:**
1. Upload blank template
2. Upload filled example
3. Click "Learn Template Structure"
4. Then upload input data

### Step 3: Process

Click **"Extract & Fill Template"**

The system will:
1. Analyze the template structure
2. Detect insertion points
3. Extract data from input files
4. Map data to placeholders using AI
5. Generate the filled document

### Step 4: Download

Review the confidence scores and download your filled document.

---

## Supported Patterns

### Explicit Placeholders (95% confidence)
```
{{client_name}}
${amount}
[CLIENT NAME]
<<<placeholder>>>
__FIELD_NAME__
```

### Dynamic Values (70-85% confidence)
```
£150,000.00          # Currency
5.5%                 # Percentage
15/03/2024           # Date
March 15, 2024       # Date (text)
john@example.com     # Email
+44 7700 900000      # Phone
```

### Context-Based (75% confidence)
```
Client Name: John Smith     # "Client Name:" context
Amount: £50,000            # "Amount:" context
Provider: Scottish Widows  # "Provider:" context
```

---

## Conditional Logic

Templer v2.0 supports conditional sections:

```
[[IF has_pension]]
This section only appears if the client has a pension.
[[/IF]]

[[IF portfolio_value > 100000]]
High-value client messaging here.
[[/IF]]

{% if risk_profile == 'aggressive' %}
Aggressive investment recommendations.
{% endif %}
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Backend** | Python 3.9+ | Core processing logic |
| **AI/LLM** | Azure OpenAI (GPT-4) | Intelligent data extraction |
| **Document** | zipfile, regex, lxml | WordML parsing |
| **PDF** | pdfplumber | PDF text extraction |

---

## Deployment Requirements

### For Streamlit Cloud

All files in `streamlit-deploy/` must be in your repository:

```
✅ templer.py
✅ requirements.txt
✅ templer_core/__init__.py
✅ templer_core/patterns.py
✅ templer_core/auto_detector.py
✅ templer_core/learning_engine.py
✅ templer_core/document_parser.py
✅ templer_core/table_processor.py
✅ templer_core/format_preserver.py
✅ templer_core/conditionals.py
✅ templer_core/validation.py
```

### Fallback Behavior

If `templer_core` is missing, the app runs in **legacy mode** (highlights only).

---

## Performance

| Metric | Value |
|--------|-------|
| Template analysis | 2-5 seconds |
| LLM mapping | 5-15 seconds |
| Document generation | 1-2 seconds |
| **Total processing** | **< 30 seconds** |

Compare to manual setup: **4 hours → 30 seconds**

---

## Confidence Scoring

Every detected field shows a confidence score:

| Score | Color | Meaning |
|-------|-------|---------|
| 70-100% | 🟢 Green | High confidence - likely correct |
| 40-69% | 🟡 Yellow | Medium confidence - review recommended |
| 0-39% | 🔴 Red | Low confidence - manual check needed |

---

## Troubleshooting

### "No insertion points found"

**Auto-Detect mode:**
- Ensure template has recognizable patterns (dates, amounts, names)
- Try adding explicit placeholders like `{{field_name}}`

**Learning mode:**
- Make sure blank and filled templates have actual differences
- Check that the filled example has real data, not placeholders

**Manual mode:**
- Highlight text in Word using the highlight tool (not text color)
- Save and re-upload the template

### "API Error"

- Check Azure OpenAI credentials in secrets
- Verify your deployment name is correct
- Ensure API key has not expired

### "Formatting broken in output"

- This is rare in v2.0 due to format preservation
- Try using a simpler template structure
- Report issue with sample files

---

## Roadmap

- [ ] Batch processing (multiple clients)
- [ ] Template version control
- [ ] Custom pattern definitions
- [ ] Direct Word editing integration
- [ ] Compliance rule validation

---

## License

MIT License

---

## Built For

**AdvisoryAI Hack-to-Hire: Template Intelligence Engine Challenge**

*The best financial advice is personalised, compliant, and clearly communicated. Templates are the vessel for that communication. Templer makes it instant.*

---

## Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/intelligent-template-builder/issues)
- **Live Demo**: [templer.streamlit.app](https://templer.streamlit.app/)

---

<div align="center">

**From 4 hours to 30 seconds. That's Templer v2.0.**

</div>
