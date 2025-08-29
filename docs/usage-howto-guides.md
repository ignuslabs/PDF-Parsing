# Usage How-To Guides

This collection of how-to guides provides step-by-step instructions for specific tasks and document processing scenarios. Each guide is problem-oriented and designed to help you accomplish particular goals efficiently.

## Document Processing How-To Guides

### How to Process Different Document Types

#### Scientific Papers and Academic Documents

**When to use**: Research papers, journal articles, conference proceedings, dissertations

**Prerequisites**:
- Documents with structured sections (Abstract, Introduction, Methods, etc.)
- Mathematical formulas and citations
- Often contain tables, figures, and references

**Steps**:

1. **Configure for Academic Content**:
   ```python
   # Launch with academic-optimized settings
   parsing_config = {
       'formula_detection': True,
       'table_recognition': True, 
       'citation_extraction': True,
       'ocr_language': 'eng+lat'  # English + Latin for scientific terms
   }
   ```

2. **Upload Document**:
   - Navigate to "📄 Parse" section
   - Upload your academic PDF
   - Select "Academic/Scientific" preset if available

3. **Verify Formula Extraction**:
   - Check "Formula" elements in results
   - Mathematical expressions should be preserved with LaTeX formatting
   - Complex equations may need manual verification

4. **Handle Tables and Figures**:
   - Review extracted table structure
   - Image elements should include captions
   - Cross-reference table/figure numbers with content

**Expected Results**:
- Structured sections (Abstract, Introduction, etc.)
- Mathematical formulas as separate elements
- Tables with preserved structure
- Figure captions and references
- Bibliography entries (if clearly formatted)

**Troubleshooting**:
- **Poor formula recognition**: Enable higher confidence thresholds
- **Missing references**: Check if bibliography is in standard format
- **Table structure lost**: Try manual table verification mode

---

#### Business Reports and Financial Documents

**When to use**: Annual reports, financial statements, business presentations, consulting reports

**Prerequisites**:
- Documents with mixed content (text, tables, charts)
- Financial data and numerical content
- Corporate formatting and layouts

**Steps**:

1. **Configure for Business Content**:
   ```python
   parsing_config = {
       'table_recognition': True,  # Critical for financial data
       'number_preservation': True,
       'header_detection': True,
       'ocr_language': 'eng'
   }
   ```

2. **Process Document**:
   - Upload business document
   - Enable "Preserve Formatting" option
   - Set confidence threshold to 85%+ for financial data

3. **Verify Financial Tables**:
   - Review all table extractions carefully
   - Check numerical accuracy in financial statements
   - Ensure currency symbols and formatting preserved

4. **Validate Headers and Structure**:
   - Confirm section headings are properly identified
   - Check executive summary extraction
   - Verify footnotes and disclaimers

**Best Practices**:
- Always verify numerical data manually
- Pay special attention to decimal points and thousands separators
- Check date formats and fiscal year references
- Review legal disclaimers for accuracy

---

#### Legal Documents and Contracts

**When to use**: Contracts, legal briefs, court documents, legislation, policy documents

**Prerequisites**:
- Structured legal formatting
- Section numbering and clause organization
- Legal terminology and citations

**Steps**:

1. **Configure for Legal Content**:
   ```python
   parsing_config = {
       'preserve_structure': True,
       'clause_detection': True,
       'legal_citation_format': True,
       'high_precision_mode': True
   }
   ```

2. **Process with High Accuracy**:
   - Use highest confidence settings (95%+)
   - Enable clause numbering preservation
   - Maintain original paragraph structure

3. **Verify Critical Sections**:
   - Review definitions and terms sections
   - Check clause numbering accuracy
   - Verify dates, names, and amounts
   - Confirm legal citations format

4. **Handle Special Formatting**:
   - Preserve indentation for subsections
   - Maintain bullet points and numbering
   - Check signature blocks and dates

**Critical Checks**:
- ✅ All dates are correctly extracted
- ✅ Names and entities are spelled correctly
- ✅ Monetary amounts match exactly
- ✅ Clause references are preserved
- ✅ Legal citations are properly formatted

---

#### Technical Manuals and Documentation

**When to use**: User manuals, technical specifications, installation guides, API documentation

**Prerequisites**:
- Step-by-step procedures
- Code snippets or technical examples
- Diagrams and technical illustrations

**Steps**:

1. **Configure for Technical Content**:
   ```python
   parsing_config = {
       'code_block_detection': True,
       'preserve_formatting': True,
       'technical_diagram_ocr': True,
       'step_numbering': True
   }
   ```

2. **Process Technical Elements**:
   - Enable code block recognition
   - Preserve monospace font formatting
   - Extract diagram labels and callouts

3. **Verify Procedural Content**:
   - Check step numbering sequences
   - Verify code examples maintain syntax
   - Ensure technical terms are preserved
   - Review diagram annotations

**Special Considerations**:
- Code snippets require manual verification
- Technical diagrams may need description enhancement
- Version numbers and model references are critical
- Safety warnings and cautions must be preserved exactly

---

#### Scanned Documents and OCR Processing

**When to use**: Old documents, photocopied papers, scanned forms, handwritten content

**Prerequisites**:
- Image-based PDFs (no selectable text)
- May have poor image quality
- Possible skew or distortion

**Steps**:

1. **Pre-processing Check**:
   ```bash
   # Check if document is text-based or image-based
   python -c "
   import PyPDF2
   with open('document.pdf', 'rb') as f:
       reader = PyPDF2.PdfReader(f)
       text = reader.pages[0].extract_text()
       if len(text.strip()) < 50:
           print('Image-based PDF - OCR required')
       else:
           print('Text-based PDF - OCR optional')
   "
   ```

2. **Configure OCR Settings**:
   ```python
   ocr_config = {
       'ocr_enabled': True,
       'language': 'eng',  # Adjust for document language
       'image_preprocessing': True,
       'deskew_correction': True,
       'noise_reduction': True
   }
   ```

3. **Quality Assessment**:
   - Review confidence scores carefully
   - Check for common OCR errors (o/0, l/1, rn/m)
   - Verify proper word spacing
   - Look for missing or extra characters

4. **Post-processing**:
   - Use spell-check on extracted text
   - Verify numbers and dates manually
   - Check proper names and technical terms
   - Review formatting preservation

**OCR Optimization Tips**:
- Higher resolution scans yield better results
- Clean, high-contrast images work best
- Avoid skewed or rotated pages
- Consider manual correction for critical content

---

### How to Configure OCR for Different Languages

#### Setting Up Multi-language Processing

**Prerequisites**:
- Tesseract OCR with language packs installed
- Documents in non-English languages
- Mixed-language documents

**Installation Check**:
```bash
# Verify available languages
tesseract --list-langs

# Should show your target languages, e.g.:
# eng
# fra
# deu  
# spa
# chi_sim
```

**Configuration Steps**:

1. **Single Language Documents**:
   ```python
   # French document
   config = {'ocr_language': 'fra'}
   
   # German document  
   config = {'ocr_language': 'deu'}
   
   # Spanish document
   config = {'ocr_language': 'spa'}
   
   # Chinese Simplified
   config = {'ocr_language': 'chi_sim'}
   ```

2. **Multi-language Documents**:
   ```python
   # English + French
   config = {'ocr_language': 'eng+fra'}
   
   # English + German + French
   config = {'ocr_language': 'eng+deu+fra'}
   
   # Auto-detect (if supported)
   config = {'ocr_language': 'auto'}
   ```

3. **Language-specific Optimizations**:
   ```python
   # For European languages
   european_config = {
       'ocr_language': 'eng+fra+deu+spa+ita',
       'character_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ',
       'preserve_accents': True
   }
   
   # For Asian languages
   asian_config = {
       'ocr_language': 'chi_sim+jpn+kor',
       'vertical_text': True,
       'character_spacing': 'wide'
   }
   ```

**Language-Specific Tips**:

**French Documents**:
- Watch for accent marks (é, è, à, ç)
- Common OCR errors: è→e, ç→c, ï→i
- Verify proper names and places

**German Documents**:
- Check umlauts (ä, ö, ü, ß)
- Compound words may be split incorrectly
- Verify capitalization (all nouns capitalized)

**Spanish Documents**:  
- Verify accent marks (á, é, í, ó, ú, ñ)
- Check inverted punctuation (¿, ¡)
- Review gender agreements in text

**Chinese Documents**:
- Use chi_sim for Simplified Chinese
- Use chi_tra for Traditional Chinese
- May require post-processing for punctuation

---

### How to Optimize Performance for Large Documents

#### Processing Multi-hundred Page Documents

**When to use**: Books, comprehensive reports, large manuals, archived documents

**Prerequisites**:
- Documents over 50 pages
- Sufficient system resources (8GB+ RAM recommended)
- Time for extended processing

**Performance Configuration**:

1. **Memory Management**:
   ```python
   large_doc_config = {
       'batch_size': 5,  # Process 5 pages at a time
       'memory_limit': 4096,  # 4GB memory limit
       'clear_cache_interval': 10,  # Clear cache every 10 pages
       'low_memory_mode': True
   }
   ```

2. **Processing Strategy**:
   ```python
   # Split processing approach
   processing_plan = {
       'chunk_size': 25,  # Process in 25-page chunks
       'parallel_processing': False,  # Avoid memory issues
       'progress_checkpoints': True,  # Save progress regularly
       'skip_images': False,  # Include images but with compression
   }
   ```

**Step-by-step Processing**:

1. **Pre-processing Assessment**:
   ```bash
   # Check document size and page count
   python -c "
   import PyPDF2
   with open('large_document.pdf', 'rb') as f:
       reader = PyPDF2.PdfReader(f)
       print(f'Pages: {len(reader.pages)}')
       print(f'File size: {os.path.getsize('large_document.pdf') / (1024*1024):.1f} MB')
   "
   ```

2. **Configure for Large Documents**:
   - Set memory limits appropriately
   - Enable progress saving
   - Use lower image resolution settings
   - Consider processing in sections

3. **Monitor Progress**:
   ```bash
   # Monitor system resources during processing
   htop  # Or Task Manager on Windows
   
   # Watch memory usage
   python -c "
   import psutil
   process = psutil.Process()
   print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
   "
   ```

4. **Optimize Settings by Content Type**:
   ```python
   # Text-heavy documents
   text_heavy_config = {
       'ocr_enabled': False,  # Skip OCR for text PDFs
       'image_processing': 'minimal',
       'table_detection': 'fast'
   }
   
   # Image-heavy documents  
   image_heavy_config = {
       'image_compression': True,
       'image_resolution': 'medium',
       'ocr_enabled': True,
       'parallel_ocr': False  # Process images sequentially
   }
   ```

**Performance Benchmarks**:
- **Text documents**: ~2-5 pages/second
- **Mixed content**: ~0.5-2 pages/second  
- **Image-heavy/OCR**: ~0.1-0.5 pages/second
- **Memory usage**: 100-500MB per 100 pages

---

#### Batch Processing Multiple Documents

**When to use**: Processing document collections, automated workflows, bulk document analysis

**Prerequisites**:
- Multiple PDF files in a directory
- Consistent processing requirements
- Sufficient storage for outputs

**Setup Batch Processing**:

1. **Create Batch Configuration**:
   ```python
   batch_config = {
       'input_directory': '/path/to/documents/',
       'output_directory': '/path/to/results/',
       'file_pattern': '*.pdf',
       'parallel_jobs': 2,  # Process 2 documents simultaneously
       'continue_on_error': True,  # Don't stop on single failures
       'generate_summary': True
   }
   ```

2. **Error Handling Strategy**:
   ```python
   error_handling = {
       'retry_failed': True,
       'max_retries': 3,
       'error_log': 'batch_processing_errors.log',
       'skip_large_files': 50,  # Skip files over 50MB
       'timeout_per_document': 3600  # 1 hour timeout per document
   }
   ```

**Batch Processing Workflow**:

1. **Prepare Document Collection**:
   ```bash
   # Organize documents
   mkdir -p batch_input batch_output batch_logs
   
   # Copy documents to process
   cp /source/documents/*.pdf batch_input/
   
   # Verify file accessibility
   ls -la batch_input/ | wc -l
   ```

2. **Configure Processing Pipeline**:
   ```python
   pipeline_config = {
       'common_settings': {
           'ocr_language': 'eng',
           'confidence_threshold': 0.8,
           'export_format': 'json'
       },
       'file_specific_rules': {
           'contract_*.pdf': {'high_precision': True},
           'report_*.pdf': {'table_focus': True},
           'manual_*.pdf': {'preserve_structure': True}
       }
   }
   ```

3. **Execute Batch Processing**:
   ```bash
   # Run batch processing script
   python batch_process.py \
       --input batch_input/ \
       --output batch_output/ \
       --config batch_config.json \
       --parallel 2 \
       --verbose
   ```

4. **Monitor and Validate Results**:
   ```bash
   # Check completion status
   find batch_output/ -name "*.json" | wc -l
   
   # Review error log
   tail -f batch_processing_errors.log
   
   # Validate random samples
   python validate_batch_results.py --sample-size 10
   ```

---

### How to Use Verification Features Effectively

#### Interactive Verification Workflow

**When to use**: Critical documents requiring high accuracy, legal documents, financial reports

**Prerequisites**:
- Parsed document with extracted elements
- Time for manual verification
- Understanding of document content

**Verification Strategy**:

1. **Prioritize Elements by Importance**:
   ```python
   verification_priority = {
       'high': ['headings', 'financial_data', 'dates', 'names'],
       'medium': ['tables', 'formulas', 'citations'],  
       'low': ['regular_text', 'footnotes']
   }
   ```

2. **Set Up Verification Session**:
   - Navigate to "✅ Verify" section
   - Load document for verification
   - Configure verification display options
   - Set up correction tracking

**Step-by-step Verification Process**:

1. **Start with High-priority Elements**:
   - Review headings and section titles first
   - Verify critical data (numbers, dates, names)
   - Check document structure accuracy

2. **Use Visual Comparison**:
   ```
   ┌─ Original PDF View ──────────────┐  ┌─ Extracted Text ────────────┐
   │ [Highlighted element in PDF]     │  │ "Quarterly Revenue: $2.5M"  │
   │ Shows actual document context    │  │ Extracted by parser         │
   └──────────────────────────────────┘  └─────────────────────────────┘
   
   Actions: [✓ Correct] [✗ Incorrect] [📝 Edit] [⏭ Skip]
   ```

3. **Make Corrections Efficiently**:
   - **Correct**: Mark obviously accurate extractions
   - **Incorrect**: Mark wrong extractions and provide correct text
   - **Edit**: Make minor corrections to mostly accurate text
   - **Skip**: Move to next element without verification

4. **Track Verification Progress**:
   ```
   📊 Verification Status
   ├── Total Elements: 156
   ├── Verified: 89 (57%)
   ├── Correct: 82 (92%)
   ├── Corrected: 7 (8%)
   ├── Remaining: 67 (43%)
   └── Estimated Time: 25 minutes
   ```

**Verification Best Practices**:

- **Focus on Critical Content**: Don't verify every word
- **Use Patterns**: If similar elements are consistently wrong, note the pattern
- **Save Frequently**: Verification state should auto-save
- **Take Breaks**: Accuracy decreases with fatigue
- **Document Issues**: Note systematic problems for future reference

**Common Verification Scenarios**:

**Financial Data Verification**:
```python
# Critical checks for financial content
financial_checks = {
    'decimal_accuracy': 'Check all monetary amounts',
    'date_formats': 'Verify fiscal periods and reporting dates', 
    'percentage_signs': 'Ensure % symbols are preserved',
    'negative_values': 'Check parentheses vs minus signs',
    'thousand_separators': 'Verify commas in large numbers'
}
```

**Technical Document Verification**:
```python
# Technical content checks
technical_checks = {
    'code_snippets': 'Preserve exact syntax and formatting',
    'version_numbers': 'Verify software versions and model numbers',
    'measurements': 'Check units and precision',
    'step_numbers': 'Ensure procedural sequence is correct',
    'cross_references': 'Verify section and figure references'
}
```

---

#### Automated Quality Checks

**When to use**: Large documents, batch processing, consistency validation

**Prerequisites**:
- Completed parsing results
- Quality criteria defined
- Automated checking rules configured

**Quality Check Configuration**:

1. **Set Up Automated Checks**:
   ```python
   quality_checks = {
       'confidence_threshold': 0.85,  # Minimum confidence for elements
       'completeness_check': True,    # Check for missing content
       'consistency_rules': True,     # Apply document-specific rules
       'format_validation': True,     # Check formatting preservation
       'statistical_analysis': True  # Analyze extraction patterns
   }
   ```

2. **Configure Document-Specific Rules**:
   ```python
   # Rules for different document types
   rules = {
       'financial_documents': {
           'require_currency_symbols': True,
           'validate_number_formats': True,
           'check_date_consistency': True
       },
       'legal_documents': {
           'preserve_clause_numbering': True,
           'validate_citations': True,
           'check_cross_references': True
       },
       'technical_manuals': {
           'preserve_code_formatting': True,
           'validate_step_sequences': True,
           'check_technical_terms': True
       }
   }
   ```

**Automated Quality Assessment**:

1. **Run Quality Checks**:
   ```python
   # Execute quality assessment
   quality_report = run_quality_checks(parsed_document, rules)
   
   # Review results
   print(f"Overall Quality Score: {quality_report.overall_score}/100")
   print(f"Elements flagged for review: {len(quality_report.flagged_elements)}")
   print(f"Suggested improvements: {len(quality_report.suggestions)}")
   ```

2. **Review Quality Report**:
   ```
   📋 Quality Assessment Report
   ├── Overall Score: 87/100
   ├── Confidence Distribution:
   │   ├── High (95-100%): 67% of elements
   │   ├── Medium (85-95%): 28% of elements  
   │   └── Low (<85%): 5% of elements
   ├── Issues Found:
   │   ├── 12 elements below confidence threshold
   │   ├── 3 potential formatting issues
   │   └── 1 missing cross-reference
   └── Recommendations:
       ├── Review low-confidence extractions
       ├── Verify table structure in pages 15-17
       └── Check mathematical formulas manually
   ```

3. **Address Quality Issues**:
   - Review elements flagged by automated checks
   - Focus verification efforts on problem areas
   - Apply corrections based on quality recommendations
   - Re-run checks after corrections

---

### How to Export in Different Formats

#### JSON Export for Data Processing

**When to use**: Data analysis, integration with other systems, programmatic processing

**Prerequisites**:
- Completed document parsing
- Understanding of JSON structure
- Target system requirements

**JSON Export Configuration**:

1. **Standard JSON Export**:
   ```python
   json_config = {
       'format': 'json',
       'indent': 2,  # Pretty printing
       'include_metadata': True,
       'include_verification': True,
       'include_confidence': True,
       'unicode_support': True
   }
   ```

2. **Custom JSON Structure**:
   ```python
   custom_json_config = {
       'schema_version': '2.0',
       'element_grouping': 'by_page',  # or 'by_type', 'flat'
       'coordinate_format': 'pdf',     # or 'pixel', 'normalized'
       'text_encoding': 'utf-8',
       'date_format': 'iso8601'
   }
   ```

**Sample JSON Output Structure**:
```json
{
  "metadata": {
    "source_path": "/path/to/document.pdf",
    "filename": "document.pdf",
    "page_count": 25,
    "total_elements": 247,
    "parsed_at": "2024-01-15T10:30:00Z",
    "parser_version": "1.0.0",
    "processing_time": 45.2,
    "verification_stats": {
      "verified": 156,
      "correct": 142,
      "corrected": 14,
      "accuracy_rate": 0.91
    }
  },
  "elements": [
    {
      "id": 1,
      "text": "Executive Summary",
      "element_type": "heading",
      "page_number": 1,
      "bbox": {
        "x0": 72.0,
        "y0": 720.0,
        "x1": 520.0,
        "y1": 745.0
      },
      "confidence": 0.982,
      "metadata": {
        "font_size": 16,
        "font_weight": "bold",
        "font_family": "Arial"
      },
      "verification_status": "correct"
    }
  ],
  "export_info": {
    "exported_at": "2024-01-15T11:15:00Z",
    "export_format": "json",
    "schema_version": "2.0"
  }
}
```

---

#### CSV Export for Spreadsheet Analysis

**When to use**: Data analysis in Excel/Google Sheets, simple data processing, reporting

**Prerequisites**:
- Tabular data analysis requirements
- Spreadsheet software compatibility
- Flattened data structure needs

**CSV Export Configuration**:

1. **Basic CSV Export**:
   ```python
   csv_config = {
       'format': 'csv',
       'delimiter': ',',
       'quote_character': '"',
       'escape_character': '\\',
       'include_headers': True,
       'flatten_metadata': True
   }
   ```

2. **Custom CSV Structure**:
   ```python
   csv_structure = {
       'columns': [
           'id', 'page_number', 'element_type', 'text', 
           'confidence', 'x0', 'y0', 'x1', 'y1',
           'verification_status', 'corrected_text'
       ],
       'max_text_length': 500,  # Truncate long text
       'handle_multiline': 'escape',  # or 'truncate', 'preserve'
       'encoding': 'utf-8-bom'  # Excel compatibility
   }
   ```

**Sample CSV Output**:
```text
id,page_number,element_type,text,confidence,x0,y0,x1,y1,verification_status
1,1,heading,"Executive Summary",0.982,72.0,720.0,520.0,745.0,correct
2,1,text,"This report presents the quarterly financial results...",0.945,72.0,680.0,520.0,715.0,correct
3,1,table,"Revenue Q1: $2.5M, Q2: $2.8M, Q3: $3.1M",0.923,100.0,600.0,480.0,650.0,corrected
```

**CSV Processing Tips**:
- Handle special characters in text fields
- Preserve numerical formatting for financial data
- Use appropriate encoding for international characters
- Consider text length limits for Excel compatibility

---

#### Markdown Export for Documentation

**When to use**: Creating documentation, web publishing, version control systems

**Prerequisites**:
- Document structure preservation needs
- Markdown processing tools
- Web or documentation platform integration

**Markdown Export Configuration**:

1. **Standard Markdown Export**:
   ```python
   markdown_config = {
       'format': 'markdown',
       'preserve_structure': True,
       'include_metadata': True,
       'generate_toc': True,  # Table of contents
       'heading_style': 'atx',  # # heading style
       'code_block_style': 'fenced'  # ``` code blocks
   }
   ```

2. **Enhanced Markdown Options**:
   ```python
   enhanced_config = {
       'math_rendering': 'latex',  # For formulas
       'table_format': 'github',   # GitHub-flavored markdown tables
       'image_handling': 'reference',  # Reference-style image links
       'footnote_style': 'inline',
       'cross_references': True
   }
   ```

**Sample Markdown Output**:
```markdown
# Document Title

## Document Information

- **Source**: document.pdf
- **Pages**: 25
- **Elements**: 247
- **Parsed**: 2024-01-15T10:30:00Z
- **Accuracy**: 91%

## Page 1

### Executive Summary

This report presents the quarterly financial results for the third quarter 
of fiscal year 2024. Key highlights include:

- Revenue growth of 15% compared to previous quarter
- Expansion into three new markets
- Successful product launch in Q3

### Financial Performance

| Quarter | Revenue | Growth |
|---------|---------|---------|
| Q1      | $2.5M   | -       |
| Q2      | $2.8M   | 12%     |
| Q3      | $3.1M   | 11%     |

**Formula**: Growth Rate = (Current - Previous) / Previous × 100

### Key Metrics

The following metrics demonstrate strong performance:

1. Customer acquisition cost decreased by 8%
2. Customer lifetime value increased by 22%
3. Monthly recurring revenue grew by 18%
```

---

#### HTML Export for Web Publishing

**When to use**: Web publishing, interactive presentations, rich formatting preservation

**Prerequisites**:
- Web server or static site hosting
- HTML/CSS knowledge for customization
- Browser compatibility requirements

**HTML Export Configuration**:

1. **Basic HTML Export**:
   ```python
   html_config = {
       'format': 'html',
       'include_css': True,
       'responsive_design': True,
       'include_metadata': True,
       'syntax_highlighting': True,  # For code blocks
       'math_rendering': 'mathjax'   # For formulas
   }
   ```

2. **Advanced HTML Options**:
   ```python
   advanced_html_config = {
       'template': 'professional',  # or 'minimal', 'academic'
       'include_navigation': True,
       'generate_index': True,
       'interactive_elements': True,
       'print_stylesheet': True,
       'mobile_optimized': True
   }
   ```

**HTML Export Features**:
- **Responsive Design**: Adapts to different screen sizes
- **Interactive Navigation**: Jump between sections and pages
- **Visual Highlighting**: Show confidence levels and verification status
- **Embedded Metadata**: Accessible via JavaScript for further processing
- **Print Optimization**: Clean printing layout
- **Accessibility**: Screen reader compatible

**Sample HTML Structure**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Document Export - Smart PDF Parser</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="export-styles.css">
</head>
<body>
    <header class="document-header">
        <h1>Document Export</h1>
        <nav class="document-nav">
            <ul>
                <li><a href="#page-1">Page 1</a></li>
                <li><a href="#page-2">Page 2</a></li>
            </ul>
        </nav>
    </header>
    
    <main class="document-content">
        <section id="metadata" class="document-metadata">
            <!-- Document information -->
        </section>
        
        <section id="page-1" class="document-page">
            <h2>Page 1</h2>
            <div class="element heading" data-confidence="0.982">
                <h3>Executive Summary</h3>
            </div>
            <div class="element text" data-confidence="0.945">
                <p>This report presents the quarterly financial results...</p>
            </div>
        </section>
    </main>
    
    <script src="export-interactions.js"></script>
</body>
</html>
```

These usage guides provide comprehensive, step-by-step instructions for accomplishing specific tasks with Smart PDF Parser. Each guide focuses on solving particular problems and achieving specific outcomes efficiently.