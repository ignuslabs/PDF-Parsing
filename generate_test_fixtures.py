#!/usr/bin/env python3
"""
Generate test fixture PDFs for the Smart PDF Parser project.
Creates 8 different PDF files targeting specific parsing behaviors.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER


def create_text_simple_pdf(filepath):
    """Create a simple text PDF with headings and paragraphs."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    title = Paragraph("Simple Text Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Section 1
    heading1 = Paragraph("1. Introduction", styles['Heading1'])
    story.append(heading1)
    story.append(Spacer(1, 6))
    
    para1 = Paragraph("""This is a simple text document designed to test basic PDF parsing capabilities. 
                      It contains multiple paragraphs with different formatting styles and hierarchical headings.""", 
                      styles['Normal'])
    story.append(para1)
    story.append(Spacer(1, 12))
    
    # Section 2
    heading2 = Paragraph("2. Content Structure", styles['Heading1'])
    story.append(heading2)
    story.append(Spacer(1, 6))
    
    subheading = Paragraph("2.1 Text Elements", styles['Heading2'])
    story.append(subheading)
    story.append(Spacer(1, 6))
    
    para2 = Paragraph("""The document structure includes various text elements such as headings, paragraphs, 
                      and subsections. This helps test the parser's ability to maintain document hierarchy.""", 
                      styles['Normal'])
    story.append(para2)
    
    doc.build(story)


def create_tables_basic_pdf(filepath):
    """Create a PDF with basic tables including headers and merged cells."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    title = Paragraph("Tables Test Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Table 1 - Simple table with headers
    heading1 = Paragraph("Table 1: Revenue Data", styles['Heading1'])
    story.append(heading1)
    story.append(Spacer(1, 6))
    
    table1_data = [
        ['Quarter', 'Revenue', 'Growth'],
        ['Q1', '$100K', '10%'],
        ['Q2', '$120K', '20%'],
        ['Q3', '$150K', '25%']
    ]
    
    table1 = Table(table1_data)
    table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table1)
    story.append(Spacer(1, 24))
    
    # Table 2 - Table with merged cells
    heading2 = Paragraph("Table 2: Project Status", styles['Heading1'])
    story.append(heading2)
    story.append(Spacer(1, 6))
    
    table2_data = [
        ['Project', 'Phase 1', 'Phase 2', 'Status'],
        ['Parser Core', 'Complete', 'In Progress', 'Active'],
        ['Search Engine', 'Complete', 'Complete', 'Done'],
        ['UI Interface', 'Pending', 'Pending', 'Not Started']
    ]
    
    table2 = Table(table2_data)
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('SPAN', (1, 0), (2, 0)),  # Merge cells
    ]))
    
    story.append(table2)
    
    doc.build(story)


def create_scanned_ocr_pdf(filepath):
    """Create a PDF that simulates scanned content (rasterized text)."""
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    
    # Title (as if scanned)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Scanned Document Sample")
    
    # Simulate slightly rotated/skewed text as in scanned docs
    c.setFont("Helvetica", 12)
    y_pos = height - 150
    
    lines = [
        "This document simulates a scanned PDF that requires OCR processing.",
        "The text extraction should work when OCR is enabled in the parser.",
        "",
        "Key testing phrases:",
        "- Revenue analysis shows positive trends",
        "- Customer satisfaction metrics improved",
        "- Implementation timeline is on track",
        "",
        "This content tests the OCR capabilities of the document parser."
    ]
    
    for line in lines:
        c.drawString(100, y_pos, line)
        y_pos -= 20
    
    # Add some "scanned" artifacts - light gray rectangles to simulate scanning noise
    c.setFillColor(colors.lightgrey)
    c.setStrokeColor(colors.lightgrey)
    c.rect(50, height - 200, 5, 100, fill=1)
    c.rect(width - 55, height - 300, 5, 150, fill=1)
    
    c.save()


def create_multicolumn_pdf(filepath):
    """Create a PDF with multi-column layout and one rotated page."""
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    
    # Page 1 - Two columns
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 50, "Multi-Column Layout Document")
    
    # Left column
    c.setFont("Helvetica", 10)
    left_x = 50
    right_x = width/2 + 20
    y_start = height - 100
    
    left_text = [
        "Left Column Content",
        "",
        "This column contains the first",
        "part of the document content.",
        "Multi-column layouts are common",
        "in academic papers and reports.",
        "",
        "The parser should maintain",
        "the correct reading order",
        "when processing this layout."
    ]
    
    y_pos = y_start
    for line in left_text:
        c.drawString(left_x, y_pos, line)
        y_pos -= 15
    
    # Right column
    right_text = [
        "Right Column Content",
        "",
        "This column contains the second",
        "part of the document content.",
        "Proper column detection is",
        "crucial for maintaining the",
        "document's logical flow.",
        "",
        "Content should be processed",
        "in the correct sequence."
    ]
    
    y_pos = y_start
    for line in right_text:
        c.drawString(right_x, y_pos, line)
        y_pos -= 15
    
    c.showPage()
    
    # Page 2 - Rotated page (landscape)
    c.setPageSize((height, width))  # Rotate page size
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, width - 50, "Rotated Page (Landscape)")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, width - 100, "This page is in landscape orientation to test rotation handling.")
    c.drawString(50, width - 130, "The parser should correctly detect and process rotated content.")
    
    c.save()


def create_images_captions_pdf(filepath):
    """Create a PDF with placeholder rectangles for images and captions."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    title = Paragraph("Document with Images and Captions", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Image 1 (simulated with colored rectangle)
    story.append(Paragraph("Figure 1: Data Visualization Chart", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Create a simple "chart" using a table to simulate an image
    chart_data = [
        ['■' * 10, '■' * 15, '■' * 8],
        ['Q1', 'Q2', 'Q3'],
    ]
    
    chart_table = Table(chart_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    chart_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 1), (-1, -1), 1, colors.black)
    ]))
    
    story.append(chart_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("Caption: This chart shows quarterly performance metrics with clear growth trends.", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Image 2 (another simulated image)
    story.append(Paragraph("Figure 2: System Architecture Diagram", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    # Simple architecture diagram simulation
    arch_data = [
        ['Frontend', '→', 'Backend', '→', 'Database'],
        ['React UI', '', 'Python API', '', 'PostgreSQL']
    ]
    
    arch_table = Table(arch_data, colWidths=[1.2*inch, 0.5*inch, 1.2*inch, 0.5*inch, 1.2*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightgreen),
        ('BACKGROUND', (4, 0), (4, -1), colors.lightyellow),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOX', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(arch_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("Caption: System architecture showing the three-tier application structure.", styles['Normal']))
    
    doc.build(story)


def create_formulas_pdf(filepath):
    """Create a PDF with mathematical formulas (text-based)."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create a style for formulas
    formula_style = ParagraphStyle(
        'Formula',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontName='Courier',
        fontSize=12,
        spaceAfter=12
    )
    
    story = []
    
    # Title
    title = Paragraph("Mathematical Formulas Document", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Introduction
    intro = Paragraph("This document contains mathematical formulas to test formula recognition capabilities.", styles['Normal'])
    story.append(intro)
    story.append(Spacer(1, 12))
    
    # Formula 1
    story.append(Paragraph("Quadratic Formula:", styles['Heading2']))
    formula1 = Paragraph("x = (-b ± √(b² - 4ac)) / 2a", formula_style)
    story.append(formula1)
    story.append(Spacer(1, 12))
    
    # Formula 2
    story.append(Paragraph("Area of a Circle:", styles['Heading2']))
    formula2 = Paragraph("A = πr²", formula_style)
    story.append(formula2)
    story.append(Spacer(1, 12))
    
    # Formula 3
    story.append(Paragraph("Pythagorean Theorem:", styles['Heading2']))
    formula3 = Paragraph("a² + b² = c²", formula_style)
    story.append(formula3)
    story.append(Spacer(1, 12))
    
    # Formula 4
    story.append(Paragraph("Einstein's Mass-Energy Equivalence:", styles['Heading2']))
    formula4 = Paragraph("E = mc²", formula_style)
    story.append(formula4)
    story.append(Spacer(1, 12))
    
    # Formula 5
    story.append(Paragraph("Normal Distribution:", styles['Heading2']))
    formula5 = Paragraph("f(x) = (1/σ√(2π)) * e^(-½((x-μ)/σ)²)", formula_style)
    story.append(formula5)
    
    doc.build(story)


def create_large_pages_pdf(filepath):
    """Create a PDF with 20+ pages of light content for performance testing."""
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title page
    title = Paragraph("Large Document - Performance Test", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    intro = Paragraph("This document contains multiple pages with light content to test parsing performance.", styles['Normal'])
    story.append(intro)
    story.append(PageBreak())
    
    # Generate 20+ pages
    for page_num in range(1, 23):
        story.append(Paragraph(f"Page {page_num}", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(f"Content for page {page_num}", styles['Heading2']))
        story.append(Spacer(1, 6))
        
        content = f"""This is page {page_num} of the large document. The content is kept light to focus on 
        parsing performance rather than content complexity. Each page contains minimal text to ensure 
        the parser can handle multi-page documents efficiently. Page {page_num} processing should be fast."""
        
        story.append(Paragraph(content, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Add a small table every few pages
        if page_num % 5 == 0:
            table_data = [
                [f'Item {i}', f'Value {i}'] for i in range(1, 4)
            ]
            table_data.insert(0, ['Item', 'Value'])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
        
        if page_num < 22:  # Don't add page break after last page
            story.append(PageBreak())
    
    doc.build(story)


def create_forms_basic_pdf(filepath):
    """Create a government-form style PDF with realistic label-value pairs for KV extraction testing."""
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    
    # Form header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 60, "APPLICATION FORM")
    
    # Set up positioning
    left_margin = 50
    column_width = (width - 120) / 2  # 120px gutter between columns
    left_col_x = left_margin
    right_col_x = left_margin + column_width + 120
    
    current_y = height - 120
    line_height = 20
    
    # Create form fields with various label-value patterns
    c.setFont("Helvetica", 12)
    
    # Row 1: Two-column layout with right-of-label placement
    c.drawString(left_col_x, current_y, "Name:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 50, current_y, "John A. Smith")
    
    c.setFont("Helvetica", 12)
    c.drawString(right_col_x, current_y, "Social Security Number:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_col_x + 150, current_y, "123-45-6789")
    
    current_y -= line_height * 1.5
    
    # Row 2: Date and phone
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Date of Birth:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 90, current_y, "01/23/1980")
    
    c.setFont("Helvetica", 12)
    c.drawString(right_col_x, current_y, "Phone Number:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_col_x + 95, current_y, "(555) 123-4567")
    
    current_y -= line_height * 2
    
    # Address section - below-label placement (multi-line value)
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Address:")
    current_y -= line_height
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 20, current_y, "123 Main Street")
    current_y -= line_height
    c.drawString(left_col_x + 20, current_y, "Apt 4B")
    current_y -= line_height
    c.drawString(left_col_x + 20, current_y, "Springfield, IL 62701")
    
    current_y -= line_height * 2
    
    # Row 3: Emergency contact information
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Emergency Contact:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 130, current_y, "Jane Smith")
    
    c.setFont("Helvetica", 12)
    c.drawString(right_col_x, current_y, "Relationship:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_col_x + 85, current_y, "Spouse")
    
    current_y -= line_height * 2
    
    # Additional fields to test various patterns
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Email Address:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 100, current_y, "john.smith@email.com")
    
    c.setFont("Helvetica", 12)
    c.drawString(right_col_x, current_y, "Zip Code:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_col_x + 70, current_y, "62701")
    
    current_y -= line_height * 1.5
    
    # Employment section
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Employer:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 70, current_y, "ABC Corporation")
    
    c.setFont("Helvetica", 12)
    c.drawString(right_col_x, current_y, "Job Title:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_col_x + 70, current_y, "Software Engineer")
    
    current_y -= line_height * 1.5
    
    # Annual income with currency format
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Annual Income:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 100, current_y, "$75,000")
    
    c.setFont("Helvetica", 12)
    c.drawString(right_col_x, current_y, "Years of Employment:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_col_x + 130, current_y, "5")
    
    current_y -= line_height * 2
    
    # References section with below-label placement
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Professional Reference:")
    current_y -= line_height
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 20, current_y, "Dr. Sarah Johnson")
    current_y -= line_height
    c.drawString(left_col_x + 20, current_y, "Manager, ABC Corporation")
    current_y -= line_height
    c.drawString(left_col_x + 20, current_y, "(555) 987-6543")
    
    current_y -= line_height * 2
    
    # Checkboxes and yes/no fields
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "U.S. Citizen:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 80, current_y, "Yes")
    
    c.setFont("Helvetica", 12)
    c.drawString(right_col_x, current_y, "Security Clearance:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_col_x + 120, current_y, "No")
    
    current_y -= line_height * 2
    
    # Signature and date section
    c.setFont("Helvetica", 12)
    c.drawString(left_col_x, current_y, "Signature Date:")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_col_x + 100, current_y, "08/29/2025")
    
    # Draw some form lines to simulate a real form
    c.setStrokeColor(colors.lightgrey)
    c.setLineWidth(0.5)
    
    # Draw underlines for some fields to simulate form fields
    form_lines = [
        (left_col_x + 50, height - 120, left_col_x + 200, height - 120),  # Name line
        (right_col_x + 150, height - 120, right_col_x + 280, height - 120),  # SSN line
        (left_col_x + 90, height - 150, left_col_x + 200, height - 150),  # DOB line
        (right_col_x + 95, height - 150, right_col_x + 220, height - 150),  # Phone line
    ]
    
    for x1, y1, x2, _ in form_lines:
        c.line(x1, y1 - 2, x2, y1 - 2)
    
    c.save()


def main():
    """Generate all test fixture PDFs."""
    fixtures_dir = "tests/fixtures"
    
    # Create fixtures directory if it doesn't exist
    if not os.path.exists(fixtures_dir):
        os.makedirs(fixtures_dir)
    
    fixtures = [
        ("text_simple.pdf", create_text_simple_pdf),
        ("tables_basic.pdf", create_tables_basic_pdf),
        ("scanned_ocr_en.pdf", create_scanned_ocr_pdf),
        ("multicolumn_rotated.pdf", create_multicolumn_pdf),
        ("images_captions.pdf", create_images_captions_pdf),
        ("formulas_snippets.pdf", create_formulas_pdf),
        ("large_pages_light.pdf", create_large_pages_pdf),
        ("forms_basic.pdf", create_forms_basic_pdf)
    ]
    
    print("Generating test fixture PDFs...")
    
    for filename, create_func in fixtures:
        filepath = os.path.join(fixtures_dir, filename)
        print(f"Creating {filename}...")
        create_func(filepath)
        print(f"✓ {filename} created successfully")
    
    print(f"\nAll {len(fixtures)} test fixtures generated in {fixtures_dir}/")


if __name__ == "__main__":
    main()