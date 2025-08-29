# Installation Tutorial

This comprehensive tutorial will guide you through setting up Smart PDF Parser from scratch. We'll cover system requirements, dependency installation, environment setup, and verification that everything works correctly.

## Overview

Smart PDF Parser is built on IBM's Docling library and requires several system-level dependencies, particularly for OCR functionality. This tutorial covers:

1. System requirements and compatibility
2. Python environment setup
3. Core dependency installation
4. Tesseract OCR installation (OS-specific)
5. Installation verification and testing
6. Common setup issues and solutions

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher (3.10+ recommended)
- **RAM**: 4GB minimum (8GB recommended for large documents)
- **Storage**: 2GB free space (includes models and dependencies)
- **Internet**: Required for initial setup and model downloads

**Recommended Setup:**
- **Python 3.10 or 3.11** for best compatibility
- **8GB+ RAM** for processing large documents
- **SSD storage** for faster model loading
- **Modern CPU** (quad-core recommended)

### Compatibility Matrix

| OS | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|----|------------|-------------|-------------|-------------|
| Windows 10+ | ✅ | ✅ | ✅ | ⚠️ Limited |
| macOS 10.14+ | ✅ | ✅ | ✅ | ⚠️ Limited |
| Ubuntu 18.04+ | ✅ | ✅ | ✅ | ⚠️ Limited |
| CentOS/RHEL 7+ | ✅ | ✅ | ✅ | ❌ Not tested |

> **Note**: Python 3.12 support is limited due to some dependencies not being fully compatible yet.

## Step 1: Python Environment Setup

### 1.1 Verify Python Installation

First, check your Python version:

```bash
# Check Python version
python --version
# or
python3 --version

# Should output: Python 3.9.x or higher
```

If you don't have Python installed or have an older version:

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer with "Add to PATH" checked
3. Choose "Install for all users" if prompted

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.10

# Using pyenv (alternative)
brew install pyenv
pyenv install 3.10.12
pyenv global 3.10.12
```

**Linux (Ubuntu/Debian):**
```bash
# Update package list
sudo apt update

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev

# Verify installation
python3.10 --version
```

### 1.2 Create Project Directory

Create a dedicated directory for Smart PDF Parser:

```bash
# Create project directory
mkdir smart-pdf-parser
cd smart-pdf-parser

# Verify you're in the correct directory
pwd
# Should show: /path/to/smart-pdf-parser
```

### 1.3 Set Up Virtual Environment

Create an isolated Python environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows (Command Prompt):
venv\Scripts\activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Verify activation - you should see (venv) in your prompt
```

### 1.4 Upgrade Pip and Core Tools

Ensure you have the latest package management tools:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install wheel and setuptools
pip install --upgrade wheel setuptools

# Verify pip version (should be 23.0+)
pip --version
```

## Step 2: Core Dependencies Installation

### 2.1 Install Requirements

Smart PDF Parser has two requirement files:

```bash
# Download/copy the requirements files to your project directory
# Then install core dependencies
pip install -r requirements.txt

# This will install:
# - docling (IBM's document parsing library)
# - streamlit (web interface)
# - pandas, numpy (data processing)
# - pillow, opencv-python (image processing)
# - fuzzywuzzy, python-levenshtein (fuzzy matching)
# - tqdm, click (utilities)
```

The installation may take 5-10 minutes as it downloads several packages.

### 2.2 Install Development Dependencies (Optional)

If you plan to contribute to development or run tests:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# This includes:
# - pytest (testing framework)
# - black, flake8 (code formatting and linting)
# - mypy (type checking)
# - memory-profiler (performance testing)
# - hypothesis (property testing)
```

### 2.3 Verify Core Installation

Test that the core libraries installed correctly:

```bash
# Test Python imports
python -c "import docling; print('Docling:', docling.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import PIL; print('Pillow: OK')"
python -c "import cv2; print('OpenCV: OK')"
```

Expected output:
```
Docling: 1.x.x
Streamlit: 1.28.x
Pandas: 2.x.x
Pillow: OK
OpenCV: OK
```

## Step 3: Tesseract OCR Installation

Tesseract is required for OCR functionality with scanned documents.

### 3.1 Windows Installation

**Method 1: Pre-compiled Installer (Recommended)**

1. Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer (tesseract-ocr-w64-setup-5.3.x.exe)
3. **Important**: During installation, note the installation path (usually `C:\Program Files\Tesseract-OCR`)
4. Add Tesseract to PATH:
   ```cmd
   # Add to system PATH
   setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
   
   # Restart command prompt and verify
   tesseract --version
   ```

**Method 2: Using Package Manager**

```powershell
# Using Chocolatey
choco install tesseract

# Using Scoop
scoop install tesseract
```

### 3.2 macOS Installation

**Method 1: Homebrew (Recommended)**

```bash
# Install Tesseract with all language packs
brew install tesseract

# Install additional languages (optional)
brew install tesseract-lang

# Verify installation
tesseract --version
tesseract --list-langs
```

**Method 2: MacPorts**

```bash
# Install base package
sudo port install tesseract

# Install language data
sudo port install tesseract-eng tesseract-osd
```

### 3.3 Linux Installation

**Ubuntu/Debian:**

```bash
# Update package list
sudo apt update

# Install Tesseract and English language pack
sudo apt install tesseract-ocr tesseract-ocr-eng

# Install additional languages (optional)
sudo apt install tesseract-ocr-fra tesseract-ocr-spa tesseract-ocr-deu

# Verify installation
tesseract --version
tesseract --list-langs
```

**CentOS/RHEL/Fedora:**

```bash
# CentOS/RHEL (with EPEL)
sudo yum install epel-release
sudo yum install tesseract tesseract-langpack-eng

# Fedora
sudo dnf install tesseract tesseract-langpack-eng

# Verify installation
tesseract --version
```

### 3.4 Verify Tesseract Installation

Test Tesseract functionality:

```bash
# Check version and available languages
tesseract --version
tesseract --list-langs

# Expected output should include:
# tesseract 5.x.x
# List of available languages:
# eng
# osd
```

Test OCR functionality:

```bash
# Create a simple test (optional)
echo "Testing Tesseract OCR" > test.txt

# If you have an image file, test OCR:
# tesseract image.png output.txt

# Test programmatic access
python -c "import tesserocr; print('TesserOCR version:', tesserocr.tesseract_version())"
```

## Step 4: Install Smart PDF Parser

### 4.1 Download Source Code

Choose one of these methods:

**Method 1: Download Project Files**
```bash
# If you have the project files, copy them to your directory
# Ensure you have these key files:
# - src/
# - requirements.txt
# - requirements-dev.txt
# - run_app.py
# - pyproject.toml
```

**Method 2: Clone Repository (if available)**
```bash
# If using git
git clone [repository-url] .
```

### 4.2 Install in Development Mode

Install the package in development mode:

```bash
# Install the package in editable mode
pip install -e .

# This creates a link to your source code, so changes are immediately available
```

### 4.3 Verify Project Structure

Your directory should look like:

```
smart-pdf-parser/
├── venv/                    # Virtual environment
├── src/                     # Source code
│   ├── core/               # Core parsing and search logic
│   ├── ui/                 # Streamlit interface
│   ├── verification/       # Verification system
│   └── utils/             # Utilities and exceptions
├── tests/                  # Test suite
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Development dependencies
├── run_app.py             # Application launcher
└── pyproject.toml         # Project configuration
```

## Step 5: First Run and Verification

### 5.1 Launch the Application

Test the complete installation:

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Launch the application
python run_app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### 5.2 Test Core Functionality

1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Check Interface**: You should see the Smart PDF Parser interface with:
   - Sidebar with four sections (Parse, Search, Verify, Export)
   - Main area showing welcome screen
   - No error messages in the browser or console

3. **Test Document Upload**: 
   - Go to the "📄 Parse" section
   - Try uploading a simple PDF (create a test PDF if needed)
   - Verify parsing completes without errors

### 5.3 Run System Tests

Test the installation with the included test suite:

```bash
# Run basic functionality tests
python -c "from src.core.parser import DoclingParser; print('Parser: OK')"
python -c "from src.core.search import SmartSearchEngine; print('Search: OK')"
python -c "from src.verification.interface import VerificationInterface; print('Verification: OK')"

# Run unit tests (if available)
pytest tests/ -v -x  # Stop on first failure

# Test with a sample document
python -c "
from src.core.parser import DoclingParser
parser = DoclingParser()
print('Parser initialized successfully!')
"
```

### 5.4 Model Download Verification

The first time you use Docling, it downloads AI models:

```bash
# Test model download (this may take several minutes on first run)
python -c "
from docling import DocumentConverter
converter = DocumentConverter()
print('Docling models ready!')
"
```

**Note**: This downloads approximately 1-2GB of AI models to `~/.cache/docling/`. This is normal and only happens once.

## Step 6: Configuration and Optimization

### 6.1 Environment Variables (Optional)

Create a `.env` file for configuration:

```bash
# Create .env file
cat > .env << 'EOF'
# Smart PDF Parser Configuration

# Tesseract Configuration
TESSERACT_CMD=/usr/local/bin/tesseract  # Adjust path as needed

# Performance Settings
MAX_WORKERS=4                           # CPU cores for parallel processing
CACHE_SIZE=100                         # Number of documents to cache
MEMORY_LIMIT=4096                      # Memory limit in MB

# UI Configuration  
STREAMLIT_PORT=8501                    # Web interface port
STREAMLIT_HOST=localhost               # Bind address

# Development Settings
DEBUG=False                            # Enable debug mode
LOG_LEVEL=INFO                         # Logging level
EOF
```

### 6.2 Verify Configuration

Test your configuration:

```bash
# Test environment loading
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Environment loaded successfully!')
"

# Test Tesseract path
python -c "
import subprocess
result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
print('Tesseract path configured correctly!')
"
```

## Installation Verification Checklist

Use this checklist to verify your installation:

### ✅ System Requirements
- [ ] Python 3.9+ installed and accessible
- [ ] Sufficient disk space (2GB+)
- [ ] Internet connection for model downloads

### ✅ Python Environment
- [ ] Virtual environment created and activated
- [ ] Pip upgraded to latest version
- [ ] Virtual environment shows in command prompt

### ✅ Core Dependencies
- [ ] `requirements.txt` installed without errors
- [ ] Docling imports successfully
- [ ] Streamlit imports successfully
- [ ] All core libraries import correctly

### ✅ Tesseract OCR
- [ ] Tesseract installed and in PATH
- [ ] `tesseract --version` works
- [ ] At least English language pack available
- [ ] TesserOCR Python binding works

### ✅ Smart PDF Parser
- [ ] Source code in correct directory structure
- [ ] Package installed in development mode
- [ ] All modules import successfully

### ✅ Application Launch
- [ ] `python run_app.py` launches without errors
- [ ] Web interface accessible at localhost:8501
- [ ] All sections (Parse, Search, Verify, Export) visible
- [ ] Can upload and process a test document

### ✅ First Document Test
- [ ] Successfully uploaded a PDF
- [ ] Parsing completed without errors
- [ ] Elements extracted and displayed
- [ ] Search functionality works
- [ ] Can navigate between sections

## Next Steps

After successful installation:

1. **Complete the Getting Started Tutorial** to learn the basic workflow
2. **Review the Troubleshooting Guide** for common issues
3. **Explore Usage How-To Guides** for specific document types
4. **Set up development environment** if contributing to the project

## Performance Optimization Tips

### For Better Performance:
1. **Use SSD storage** for faster model loading
2. **Increase RAM allocation** for large documents
3. **Adjust worker processes** based on CPU cores
4. **Pre-load models** in production environments
5. **Use document caching** for repeated processing

### Memory Management:
```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Troubleshooting

If you encounter issues during installation, check:

1. **Python Version**: Ensure you're using Python 3.9+
2. **Virtual Environment**: Make sure it's activated
3. **Internet Connection**: Required for downloading dependencies and models
4. **Disk Space**: Ensure sufficient space for models and cache
5. **Permissions**: Some installations may require administrator/sudo rights

For detailed troubleshooting, see the **Installation Troubleshooting How-To Guide**.

## Success!

If you've completed all steps and passed the verification checklist, you have successfully installed Smart PDF Parser! The system is ready for document processing, and you can proceed with the Getting Started Tutorial to learn the core workflow.