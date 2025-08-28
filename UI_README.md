# 📄 Smart PDF Parser - User Interface

A comprehensive Streamlit-based user interface for the Smart PDF Parser, providing visual document parsing, search, verification, and export capabilities.

## 🚀 Quick Start

### 1. Launch the Application

```bash
# From the project root directory
python run_app.py
```

The application will start at `http://localhost:8501`

### 2. Basic Workflow

1. **📄 Parse** - Upload and parse PDF documents
2. **🔍 Search** - Find content across documents  
3. **✅ Verify** - Visually verify and correct parsed elements
4. **📊 Export** - Download results in various formats

## 🎯 Features

### Document Parsing Page (📄 Parse)
- **PDF Upload**: Drag & drop or browse for PDF files
- **Configuration**: OCR settings, table extraction, image generation
- **Presets**: Quick configuration for different document types
- **Progress Tracking**: Real-time parsing progress
- **Results Preview**: Immediate parsing results display

### Search Page (🔍 Search)
- **Multi-Mode Search**: Exact, fuzzy, and semantic search
- **Advanced Filters**: Element type, page range, confidence threshold
- **Result Ranking**: Relevance scoring and sorting
- **Navigation**: Jump directly to verification from search results
- **Analytics**: Search result statistics and insights

### Verification Page (✅ Verify)
- **Visual Overlays**: PDF pages with element bounding boxes
- **Interactive Selection**: Click on elements to select them
- **Element Details**: Full text, metadata, and position info
- **Verification Actions**: Mark correct/incorrect/partially correct
- **Progress Tracking**: Real-time verification progress
- **Bulk Operations**: Process multiple elements at once

### Export Page (📊 Export)
- **Multiple Formats**: JSON, CSV, Markdown, HTML, Plain Text
- **Content Options**: Metadata, bounding boxes, verification data
- **Preview**: Live preview of export files
- **Download**: Individual files or ZIP archives
- **Analytics**: Document and element statistics

## 🏗️ Architecture

### Directory Structure
```
src/ui/
├── app.py                    # Main application entry point
├── pages/                    # Streamlit pages
│   ├── 1_📄_Parse.py        # Document parsing page
│   ├── 2_🔍_Search.py       # Search interface page
│   ├── 3_✅_Verify.py       # Verification page
│   └── 4_📊_Export.py       # Export page
├── components/               # Reusable UI components
│   ├── upload_handler.py    # PDF upload component
│   ├── config_panel.py      # Configuration panel
│   └── results_display.py   # Results display component
├── utils/                   # Utility functions
│   ├── state_manager.py     # Session state management
│   └── export_handler.py    # Export utilities
└── styles.py               # Custom CSS styling
```

### Key Components

#### StateManager
- Manages Streamlit session state
- Provides validation and callbacks
- Handles state persistence across pages

#### PDFUploadHandler
- Validates PDF files
- Manages temporary file creation
- Provides upload progress feedback

#### ResultsDisplay
- Displays parsing results with filtering
- Shows search results with highlighting
- Provides element selection interface

#### ExportHandler
- Handles multiple export formats
- Creates download links and ZIP files
- Provides file previews

## 🎨 Visual Design

### Color Scheme
- **Primary**: Gradient blues (`#667eea` to `#764ba2`)
- **Secondary**: Warm gradients for accents
- **Element Types**: Color-coded badges
- **Status Indicators**: Green/Yellow/Red for confidence and verification

### Interactive Elements
- **Hover Effects**: Subtle animations and transitions
- **Progress Indicators**: Visual progress bars and metrics
- **Responsive Layout**: Adapts to different screen sizes
- **Custom Scrollbars**: Styled scrollbars matching theme

### Element Type Colors
- **Text**: Blue (`#2E86AB`)
- **Heading**: Purple (`#A23B72`) 
- **Table**: Orange (`#F18F01`)
- **List**: Red (`#C73E1D`)
- **Image**: Green (`#0F7B0F`)
- **Caption**: Dark Purple (`#7209B7`)
- **Formula**: Pink (`#F72585`)
- **Code**: Light Blue (`#4361EE`)

## 🔧 Configuration

### Parser Settings
- **OCR Options**: Enable/disable, engine selection, language settings
- **Table Extraction**: Mode selection (accurate/fast)
- **Image Generation**: Page images for verification
- **Performance**: Memory optimization settings

### Presets Available
- **Fast Processing**: Quick parsing with basic features
- **Standard Processing**: Balanced speed and accuracy
- **High Accuracy**: Best quality with OCR enabled
- **Scanned Documents**: Optimized for scanned PDFs
- **Verification Mode**: Full features for verification workflow

## 📱 Usage Guide

### Parsing Documents

1. Navigate to **📄 Parse** page
2. Configure parser settings or choose a preset
3. Upload PDF files (drag & drop supported)
4. Click "🚀 Parse Documents"
5. Review parsing results and element breakdown

### Searching Content

1. Navigate to **🔍 Search** page  
2. Enter search query
3. Select search mode (exact/fuzzy/semantic)
4. Apply filters if needed
5. Click on results to navigate to verification

### Verifying Elements

1. Navigate to **✅ Verify** page
2. Select document and navigate pages
3. Click on highlighted elements in PDF view
4. Review element details in sidebar
5. Mark elements as correct/incorrect/partial
6. Use bulk actions for efficiency

### Exporting Results

1. Navigate to **📊 Export** page
2. Select export formats and content options
3. Choose documents to include
4. Click "🚀 Generate Export Files"
5. Preview files and download individually or as ZIP

## 🔒 Session Management

### State Persistence
- Document data persists across page navigation
- Parser configuration remembered
- Search history maintained
- Verification progress tracked

### Memory Management
- Automatic cleanup of temporary files
- Efficient handling of large documents
- Progress indicators for memory-intensive operations

## 🐛 Troubleshooting

### Common Issues

**Application won't start:**
- Check Python dependencies: `pip install -r requirements.txt`
- Verify you're in the project root directory
- Check port 8501 is available

**PDF upload fails:**
- Verify file is a valid PDF (not corrupted)
- Check file size (max 100MB per file)
- Ensure sufficient disk space

**Parsing is slow:**
- Disable OCR if not needed
- Use "Fast Processing" preset
- Reduce image scale factor
- Process smaller documents first

**Out of memory errors:**
- Use "Fast Processing" preset
- Disable page image generation
- Process documents one at a time
- Reduce image scale factor

### Performance Tips

1. **For Large Documents:**
   - Disable OCR unless needed
   - Use fast table mode
   - Set image scale to 0.8 or lower

2. **For Scanned Documents:**
   - Enable OCR
   - Use EasyOCR for better accuracy
   - Increase image scale to 1.2-1.5

3. **For Verification:**
   - Always enable page image generation
   - Use standard or verification preset
   - Process in smaller batches

## 🔄 Updates & Development

### Adding New Features

1. **New Pages**: Add to `src/ui/pages/` with proper naming
2. **Components**: Add to `src/ui/components/` and update `__init__.py`
3. **Styles**: Update `src/ui/styles.py` for visual consistency
4. **State**: Use `StateManager` for proper state handling

### Custom Styling

Modify `src/ui/styles.py` to:
- Change color scheme
- Add new element type colors
- Customize animations
- Update responsive breakpoints

## 📊 Analytics & Metrics

The application tracks:
- Document parsing statistics
- Element type distributions
- Verification progress
- Search result relevance
- Export format usage

## 🤝 Integration

### With Core Components
- **DoclingParser**: Document parsing engine
- **SmartSearchEngine**: Multi-modal search
- **VerificationInterface**: Element verification
- **PDFRenderer**: Visual overlay rendering

### External Dependencies
- **Streamlit**: Web application framework
- **Docling**: PDF processing engine
- **PIL/Pillow**: Image processing
- **Pandas**: Data manipulation
- **JSON/CSV**: Export formats

## 📝 License

This UI system is part of the Smart PDF Parser project and follows the same licensing terms.