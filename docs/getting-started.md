# Getting Started Tutorial

Welcome to Smart PDF Parser! This tutorial will guide you through your first experience with the application, from initial setup to processing your first PDF document. By the end of this tutorial, you'll understand the core workflow and have successfully parsed, searched, and verified content from a PDF.

## Prerequisites

Before we begin, ensure you have:

- **Python 3.9 or higher** installed on your system
- **A PDF document** ready for testing (we recommend starting with a simple text-based PDF)
- **Basic familiarity** with command line operations
- **10-15 minutes** to complete this tutorial

## What You'll Learn

In this tutorial, you will:

1. Set up the Smart PDF Parser environment
2. Launch the application interface
3. Parse your first PDF document
4. Explore the extracted elements
5. Perform basic search operations
6. Use the verification system
7. Export your results

## Step 1: Quick Setup

### 1.1 Download and Extract

First, download the Smart PDF Parser and navigate to the project directory:

```bash
# Navigate to your project directory
cd /path/to/PDF\ Parsing

# Verify you're in the correct directory
ls -la
# You should see: src/, tests/, requirements.txt, run_app.py
```

### 1.2 Create Python Environment

Create a dedicated Python environment for the project:

```bash
# Create virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify activation - you should see (venv) in your prompt
```

### 1.3 Install Core Dependencies

Install the required packages:

```bash
# Install core requirements
pip install -r requirements.txt

# Verify Docling installation (this may take a few minutes)
python -c "import docling; print('Docling installed successfully!')"
```

> **Note**: The first time you import Docling, it may download additional models. This is normal and only happens once.

## Step 2: Launch the Application

### 2.1 Start the Interface

Launch the Streamlit web interface:

```bash
# Start the application
python run_app.py
```

You should see output similar to:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.xxx:8501
```

### 2.2 Access the Interface

1. Open your web browser
2. Navigate to `http://localhost:8501`
3. You should see the Smart PDF Parser welcome screen with a sidebar containing four main sections:
   - 📄 **Parse** - Document processing
   - 🔍 **Search** - Content search and filtering  
   - ✅ **Verify** - Interactive verification
   - 📊 **Export** - Results export

## Step 3: Parse Your First Document

### 3.1 Navigate to Parse Section

1. Click on "📄 Parse" in the sidebar
2. You'll see the document parsing interface with upload options

### 3.2 Upload a PDF Document

1. Click the "Browse files" button or drag and drop a PDF
2. For your first attempt, choose a **simple text-based PDF** (avoid scanned documents initially)
3. Recommended test document characteristics:
   - Less than 10 pages
   - Contains mostly text
   - Has some headings or structure
   - No complex tables or images initially

### 3.3 Configure Parsing Options

Before parsing, you can adjust settings:

```
📋 Parsing Configuration
├── OCR Language: English (default)
├── Table Recognition: ✓ Enabled  
├── Formula Detection: ✓ Enabled
└── Image Extraction: ✓ Enabled
```

> **Tip**: For your first document, keep the default settings.

### 3.4 Start Parsing

1. Click the "Parse Document" button
2. You'll see a progress indicator showing:
   ```
   🔄 Processing document...
   ├── Loading document: ✓ Complete
   ├── Extracting elements: ⏳ In progress
   └── Building structure: ⏳ Pending
   ```

3. Parsing typically takes 10-30 seconds for small documents

### 3.5 Review Parsing Results

Once complete, you'll see:

**Document Summary:**
```
📊 Parsing Results
├── Total Elements: 47
├── Pages Processed: 3
├── Text Elements: 32
├── Headings: 8
├── Tables: 2
├── Images: 5
└── Average Confidence: 94.2%
```

**Element Preview:**
The interface displays the first few extracted elements with their types, confidence scores, and preview text.

## Step 4: Explore Your Document

### 4.1 Browse Elements by Type

Use the element type filter to explore different content:

1. **Text Elements**: Regular paragraphs and content
2. **Headings**: Section titles and headers  
3. **Tables**: Structured data (if present)
4. **Images**: Visual content descriptions
5. **Formulas**: Mathematical expressions (if present)

### 4.2 Navigate by Page

Use the page navigation controls:

```
Page Navigation: [◀ Prev] Page 1 of 3 [Next ▶]
```

Click through pages to see how elements are distributed across your document.

### 4.3 Examine Element Details

Click on any element to view detailed information:

```
📋 Element Details
├── Text: "Introduction to Machine Learning"
├── Type: heading
├── Page: 1
├── Confidence: 98.5%
├── Position: (72.0, 720.5, 523.2, 745.8)
└── Metadata: {"font_size": 14, "font_weight": "bold"}
```

## Step 5: Perform Your First Search

### 5.1 Navigate to Search Section

1. Click on "🔍 Search" in the sidebar
2. You'll see the search interface with various options

### 5.2 Try Basic Text Search

1. **Exact Search**: Type a phrase you know exists in your document
   ```
   Search Query: "machine learning"
   Search Type: ○ Exact ● Fuzzy ○ Semantic
   ```

2. Click "Search" and review results:
   ```
   🔍 Search Results (3 matches)
   
   Match 1: [heading] "Introduction to Machine Learning"
   ├── Score: 100% (Exact match)
   ├── Page: 1
   └── Context: "...overview of machine learning concepts..."
   
   Match 2: [text] "Machine learning algorithms can be..."
   ├── Score: 95% (Partial match)
   ├── Page: 2  
   └── Context: "...various machine learning techniques..."
   ```

### 5.3 Experiment with Search Types

Try the different search modes:

1. **Fuzzy Search**: Finds similar text even with typos
   ```
   Query: "machne lerning" (intentional typos)
   Results: Still finds "machine learning" content
   ```

2. **Semantic Search**: Finds conceptually related content
   ```
   Query: "artificial intelligence"
   Results: May find "machine learning", "neural networks", "AI" content
   ```

### 5.4 Use Search Filters

Refine your search with filters:

```
🔧 Search Filters
├── Element Types: ☑ Text ☑ Headings ☐ Tables ☐ Images
├── Page Range: 1 to 3
├── Confidence Threshold: 80%
└── Max Results: 10
```

## Step 6: Verify Content Accuracy

### 6.1 Navigate to Verify Section

1. Click on "✅ Verify" in the sidebar
2. Select an element to verify from the search results or element list

### 6.2 Visual Verification

The verification interface shows:

1. **Original Document View**: PDF page with highlighted element
2. **Extracted Text**: What the parser extracted
3. **Verification Controls**: Buttons to mark as correct/incorrect

```
🔍 Element Verification
┌─ Original Document ─────────────┐
│ [PDF page with highlighted box] │
│ showing the selected element    │
└─────────────────────────────────┘

📝 Extracted Text:
"Introduction to Machine Learning"

✅ Verification Actions:
[✓ Correct] [✗ Incorrect] [📝 Edit] [⏭ Skip]
```

### 6.3 Verify Several Elements

1. Mark obviously correct elements as "✓ Correct"
2. For incorrect extractions, click "✗ Incorrect" and provide corrections
3. Use "📝 Edit" to make minor text corrections
4. Click "⏭ Skip" to move to the next element

### 6.4 Track Verification Progress

Monitor your progress:

```
📊 Verification Progress
├── Total Elements: 47
├── Verified: 12 (25.5%)
├── Correct: 10 (83.3%)
├── Incorrect: 2 (16.7%)
└── Remaining: 35
```

## Step 7: Export Your Results

### 7.1 Navigate to Export Section

1. Click on "📊 Export" in the sidebar
2. Choose your export format and options

### 7.2 Configure Export Settings

```
📤 Export Configuration
├── Format: ● JSON ○ CSV ○ Markdown ○ HTML
├── Include Metadata: ✓ Enabled
├── Include Verification: ✓ Enabled
├── Filter by Status: ● All ○ Verified only ○ Unverified only
└── Element Types: ✓ All selected
```

### 7.3 Generate Export

1. Click "Generate Export"
2. Preview the export content in the interface
3. Click "Download" to save the file

### 7.4 Examine Export Content

Your JSON export will contain:

```json
{
  "metadata": {
    "source_path": "/path/to/your/document.pdf",
    "page_count": 3,
    "total_elements": 47,
    "parsed_at": "2024-01-15T10:30:00Z",
    "verification_stats": {
      "verified": 12,
      "correct": 10,
      "incorrect": 2
    }
  },
  "elements": [
    {
      "text": "Introduction to Machine Learning",
      "element_type": "heading",
      "page_number": 1,
      "bbox": {"x0": 72.0, "y0": 720.5, "x1": 523.2, "y1": 745.8},
      "confidence": 0.985,
      "metadata": {"font_size": 14, "font_weight": "bold"},
      "verification_status": "correct"
    }
  ]
}
```

## Step 8: Understanding the Workflow

### 8.1 Complete Workflow Summary

You've now experienced the full Smart PDF Parser workflow:

```
1. Parse 📄 → Extract elements from PDF using Docling
2. Search 🔍 → Find specific content with multiple search modes  
3. Verify ✅ → Validate extraction accuracy with visual verification
4. Export 📊 → Output results in multiple formats
```

### 8.2 Best Practices Learned

From this tutorial, remember these key practices:

1. **Start Simple**: Begin with text-based PDFs before trying complex documents
2. **Verify Systematically**: Check extraction accuracy, especially for important content
3. **Use Appropriate Search**: Choose exact/fuzzy/semantic search based on your needs
4. **Export Regularly**: Save your work frequently, especially after verification

### 8.3 Common Element Types You've Seen

- **Text**: Regular paragraph content
- **Heading**: Section titles and headers
- **Table**: Structured data in rows/columns
- **Image**: Visual content (with descriptions)
- **Formula**: Mathematical expressions
- **List**: Bulleted or numbered items

## Next Steps

### Immediate Next Actions

1. **Try Different Documents**: Test with various PDF types (reports, articles, forms)
2. **Explore Advanced Features**: Experiment with OCR settings and language options
3. **Learn Troubleshooting**: Review common issues in the How-To guides
4. **Understand Performance**: Check the performance optimization guides

### Recommended Learning Path

After completing this tutorial:

1. **Installation Tutorial** - Learn detailed setup and troubleshooting
2. **Usage How-To Guides** - Master specific use cases and document types
3. **Development How-To Guides** - Learn to extend and customize the system

### Getting Help

If you encounter issues:

1. **Check the troubleshooting section** in the Installation How-To guide
2. **Review error messages** - they often contain helpful information
3. **Verify your environment** - ensure all dependencies are properly installed
4. **Test with simpler documents** first to isolate issues

## Congratulations!

You've successfully completed your first Smart PDF Parser session! You now understand:

- ✅ How to set up and launch the application
- ✅ The basic parsing workflow and options
- ✅ How to search content with different methods  
- ✅ How to verify extraction accuracy
- ✅ How to export results in multiple formats
- ✅ The core concepts and best practices

You're ready to start processing your own documents and exploring the more advanced features covered in the How-To guides.