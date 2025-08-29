# Installation Troubleshooting How-To Guide

This guide addresses common installation issues, provides step-by-step solutions, and includes a comprehensive Tesseract OCR troubleshooting section. Use this guide when you encounter problems during installation or setup.

## Quick Diagnostic

Before diving into specific issues, run this diagnostic to identify the problem area:

```bash
# 1. Check Python version
python --version

# 2. Check virtual environment
echo $VIRTUAL_ENV  # Should show path to venv

# 3. Test core imports
python -c "import sys; print('Python path:', sys.executable)"
python -c "import docling; print('Docling: OK')" 2>/dev/null || echo "Docling: FAILED"
python -c "import streamlit; print('Streamlit: OK')" 2>/dev/null || echo "Streamlit: FAILED"
python -c "import tesserocr; print('TesserOCR: OK')" 2>/dev/null || echo "TesserOCR: FAILED"

# 4. Check Tesseract
tesseract --version 2>/dev/null || echo "Tesseract: NOT IN PATH"

# 5. Test application launch
python -c "from src.core.parser import DoclingParser; print('Parser: OK')" 2>/dev/null || echo "Parser: FAILED"
```

Based on the results, jump to the relevant section below.

## Python Environment Issues

### Problem: Wrong Python Version

**Symptoms:**
```bash
$ python --version
Python 2.7.x
# or
Python 3.8.x
```

**Solution:**

**Windows:**
```bash
# Check if Python 3.9+ is installed under different name
python3 --version
python3.9 --version
python3.10 --version

# If not found, download and install from python.org
# Then use specific version:
python3.10 -m venv venv
```

**macOS:**
```bash
# Install using Homebrew
brew install python@3.10

# Create alias (optional)
echo 'alias python3=/opt/homebrew/bin/python3.10' >> ~/.zshrc
source ~/.zshrc

# Create virtual environment with specific version
/opt/homebrew/bin/python3.10 -m venv venv
```

**Linux:**
```bash
# Ubuntu/Debian - install specific version
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Use specific version
python3.10 -m venv venv
```

### Problem: Virtual Environment Not Activated

**Symptoms:**
- Command prompt doesn't show `(venv)`
- `pip list` shows system packages instead of project packages
- Import errors for project dependencies

**Solution:**
```bash
# Check current environment
echo $VIRTUAL_ENV
# Should show: /path/to/your/project/venv

# If empty, activate virtual environment:
# macOS/Linux:
source venv/bin/activate

# Windows Command Prompt:
venv\Scripts\activate

# Windows PowerShell:
venv\Scripts\Activate.ps1

# Verify activation - prompt should show (venv)
```

### Problem: Virtual Environment Creation Fails

**Symptoms:**
```bash
$ python -m venv venv
Error: Unable to create virtual environment
```

**Solutions:**

**Missing venv module:**
```bash
# Ubuntu/Debian
sudo apt install python3-venv

# CentOS/RHEL/Fedora
sudo yum install python3-venv
# or
sudo dnf install python3-venv
```

**Permission issues:**
```bash
# Ensure you have write permissions
ls -la
# If directory is read-only, change location or fix permissions

# Alternative location
mkdir ~/smart-pdf-parser
cd ~/smart-pdf-parser
python -m venv venv
```

**Corrupted Python installation:**
```bash
# Reinstall Python (macOS with Homebrew)
brew uninstall python@3.10
brew install python@3.10

# Windows - reinstall from python.org
# Linux - reinstall python3-venv package
```

## Dependency Installation Issues

### Problem: pip Install Failures

**Symptoms:**
```bash
ERROR: Could not install packages due to an EnvironmentError
ERROR: Failed building wheel for [package]
```

**Solutions:**

**Upgrade pip and build tools:**
```bash
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade pip setuptools wheel
```

**Install system dependencies (Linux):**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential python3-dev libffi-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel libffi-devel

# Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel libffi-devel
```

**Network/proxy issues:**
```bash
# If behind corporate firewall/proxy
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Use different index
pip install -i https://pypi.org/simple/ -r requirements.txt
```

**Clean install:**
```bash
# Clear pip cache
pip cache purge

# Remove and recreate virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Problem: Docling Installation Fails

**Symptoms:**
```bash
ERROR: Could not install docling
ERROR: Microsoft Visual C++ 14.0 is required (Windows)
ERROR: Failed to build docling
```

**Solutions:**

**Windows - Missing Visual C++ Build Tools:**
```bash
# Download and install Microsoft C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# OR install Visual Studio Community with C++ support

# Alternative - use pre-compiled wheels
pip install --only-binary=all docling
```

**macOS - Missing Xcode Command Line Tools:**
```bash
# Install Xcode command line tools
xcode-select --install

# If that fails, install full Xcode from App Store
# Then retry docling installation
```

**Linux - Missing dependencies:**
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev libxml2-dev libxslt-dev libssl-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ python3-devel libxml2-devel libxslt-devel openssl-devel

# Retry installation
pip install docling
```

### Problem: Memory Issues During Installation

**Symptoms:**
```bash
Killed
MemoryError
ERROR: Operation ran out of memory
```

**Solutions:**

**Increase swap space (Linux):**
```bash
# Create temporary swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install packages
pip install -r requirements.txt

# Remove swap file after installation
sudo swapoff /swapfile
sudo rm /swapfile
```

**Install packages individually:**
```bash
# Install one at a time to reduce memory pressure
pip install docling
pip install streamlit
pip install pandas
# ... continue with other packages
```

**Use pip's no-cache option:**
```bash
pip install --no-cache-dir -r requirements.txt
```

## Tesseract OCR Troubleshooting

### Tesseract Q&A Section

#### Q: "tesseract: command not found" error

**A: Tesseract not in PATH**

**Windows:**
```batch
# Find Tesseract installation
dir "C:\Program Files\Tesseract-OCR\tesseract.exe"

# Add to PATH permanently
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"

# Or set for current session only
set PATH=%PATH%;C:\Program Files\Tesseract-OCR

# Verify
tesseract --version
```

**macOS:**
```bash
# Check if installed via Homebrew
brew list tesseract

# If installed but not in PATH
export PATH="/opt/homebrew/bin:$PATH"
# Add to ~/.zshrc for permanent effect

# Reinstall if necessary
brew uninstall tesseract
brew install tesseract
```

**Linux:**
```bash
# Find Tesseract location
which tesseract
whereis tesseract

# If installed but not found
sudo apt install tesseract-ocr

# Check PATH
echo $PATH
# Ensure /usr/bin is in PATH
```

#### Q: TesserOCR Python package fails to install

**A: Missing development headers**

**Windows:**
```bash
# Install pre-compiled wheel
pip install tesserocr --find-links https://github.com/simonflueckiger/tesserocr-windows_build/releases

# Or use alternative OCR package
pip uninstall tesserocr
pip install easyocr
```

**macOS:**
```bash
# Install dependencies first
brew install tesseract leptonica pkg-config

# Set environment variables
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
export LDFLAGS="-L/opt/homebrew/lib"
export CPPFLAGS="-I/opt/homebrew/include"

# Install tesserocr
pip install tesserocr
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev

# CentOS/RHEL/Fedora
sudo yum install tesseract-devel leptonica-devel
# or
sudo dnf install tesseract-devel leptonica-devel

# Install Python package
pip install tesserocr
```

#### Q: "Failed to init API" error when using TesserOCR

**A: Language data missing or corrupted**

```bash
# Check available languages
tesseract --list-langs

# If English (eng) is missing:
# Windows: Reinstall with language pack
# macOS:
brew reinstall tesseract tesseract-lang

# Linux:
sudo apt install tesseract-ocr-eng tesseract-ocr-osd

# Test manually
echo "test" | tesseract stdin stdout
```

#### Q: Tesseract works in command line but not in Python

**A: Path configuration issues**

```python
# Test Python access
import tesserocr
print("TesserOCR version:", tesserocr.tesseract_version())

# If that fails, try setting path explicitly
import os
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
# Adjust path based on your system

# For Windows
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
```

#### Q: OCR results are poor quality

**A: Configuration and preprocessing needed**

```python
# Configure Tesseract for better results
from PIL import Image
import tesserocr

# Load and preprocess image
image = Image.open('document.png')
# Convert to grayscale
image = image.convert('L')
# Increase contrast
from PIL import ImageEnhance
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2.0)

# Use specific OCR config
config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
text = tesserocr.image_to_text(image, config=config)
```

#### Q: Multiple language installation

**A: Installing additional language packs**

**Windows:**
```bash
# Download language files from GitHub
# https://github.com/tesseract-ocr/tessdata
# Place .traineddata files in C:\Program Files\Tesseract-OCR\tessdata\

# Verify
tesseract --list-langs
```

**macOS:**
```bash
# Install language pack
brew install tesseract-lang

# Or specific languages
brew install tesseract tesseract-lang --with-training-tools

# Verify
tesseract --list-langs
```

**Linux:**
```bash
# Install multiple languages
sudo apt install tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa tesseract-ocr-ita

# List available language packages
apt search tesseract-ocr-

# Verify installation
tesseract --list-langs
```

## Smart PDF Parser Specific Issues

### Problem: Import Errors for Smart PDF Parser Modules

**Symptoms:**
```python
ModuleNotFoundError: No module named 'src.core'
ImportError: attempted relative import with no known parent package
```

**Solutions:**

**Install in development mode:**
```bash
# From project root directory
pip install -e .

# Verify installation
pip list | grep smart-pdf-parser
```

**Check PYTHONPATH:**
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/smart-pdf-parser"

# Or in Python script
import sys
sys.path.insert(0, '/path/to/smart-pdf-parser')
```

**Verify directory structure:**
```bash
# Ensure you have the correct structure
ls -la src/
# Should show: core/, ui/, verification/, utils/

# Check for __init__.py files
find src/ -name "__init__.py"
```

### Problem: Streamlit App Won't Start

**Symptoms:**
```bash
$ python run_app.py
ModuleNotFoundError: No module named 'streamlit'
streamlit: command not found
```

**Solutions:**

**Verify Streamlit installation:**
```bash
# Check if installed
pip list | grep streamlit

# If not installed
pip install streamlit

# Test Streamlit directly
streamlit hello
```

**Port already in use:**
```bash
# Check what's using port 8501
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Use different port
streamlit run run_app.py --server.port 8502

# Or kill existing process
kill -9 [PID]  # macOS/Linux
taskkill /PID [PID] /F  # Windows
```

**Permission issues:**
```bash
# Run with different permissions
sudo python run_app.py  # Not recommended

# Better - change port to higher number
streamlit run run_app.py --server.port 9501
```

### Problem: Model Download Failures

**Symptoms:**
```bash
ConnectionError: Unable to download model
URLError: urlopen error
HTTPSConnectionPool: Max retries exceeded
```

**Solutions:**

**Network connectivity:**
```bash
# Test internet connection
ping google.com

# Test specific model servers
curl -I https://huggingface.co/

# Use proxy if behind firewall
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

**Manual model download:**
```bash
# Create cache directory
mkdir -p ~/.cache/docling/models

# Download models manually (URLs may vary)
# Check Docling documentation for current model URLs

# Set offline mode (if supported)
export DOCLING_CACHE_DIR=~/.cache/docling
```

## Performance Issues

### Problem: Slow Processing

**Symptoms:**
- Document parsing takes extremely long
- High CPU or memory usage
- Application becomes unresponsive

**Solutions:**

**Monitor resource usage:**
```bash
# Check CPU and memory usage
top -p $(pgrep -f "streamlit\|python")

# Or use htop for better visualization
htop
```

**Optimize settings:**
```python
# Reduce worker processes in .env file
MAX_WORKERS=2  # Instead of 4 on slower machines
MEMORY_LIMIT=2048  # Reduce if low RAM

# Process smaller documents first
# Disable OCR for text-based PDFs
# Use lower resolution settings
```

**Clear cache:**
```bash
# Clear Docling cache
rm -rf ~/.cache/docling/

# Clear pip cache
pip cache purge

# Restart application
```

## System-Specific Issues

### Windows Specific

**Problem: PowerShell execution policy**
```powershell
# Error: execution of scripts is disabled
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate virtual environment
venv\Scripts\Activate.ps1
```

**Problem: Long path names**
```batch
# Enable long paths in Windows
# Run as Administrator:
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1

# Or use shorter paths
cd C:\
mkdir spp
cd spp
```

### macOS Specific

**Problem: SSL certificate issues**
```bash
# Update certificates
/Applications/Python\ 3.x/Install\ Certificates.command

# Or install certificates via Homebrew
brew install ca-certificates
```

**Problem: Apple Silicon compatibility**
```bash
# For M1/M2 Macs, some packages may need Rosetta
# Check if running under Rosetta
arch

# Install Rosetta if needed
softwareupdate --install-rosetta
```

### Linux Specific

**Problem: Missing system libraries**
```bash
# Common missing libraries
sudo apt install libgl1-mesa-glx  # For OpenCV
sudo apt install libglib2.0-0     # For various packages
sudo apt install libblas3 liblapack3 liblapack-dev libblas-dev  # For NumPy/SciPy
```

## Getting Additional Help

### Enable Debug Mode

```bash
# Create .env file with debug settings
cat > .env << 'EOF'
DEBUG=True
LOG_LEVEL=DEBUG
STREAMLIT_DEBUG=True
EOF

# Run with verbose output
python run_app.py --verbose
```

### Collect System Information

```bash
# Create diagnostic report
cat > diagnostic.txt << 'EOF'
=== System Information ===
OS: $(uname -a 2>/dev/null || echo "Windows")
Python: $(python --version)
Pip: $(pip --version)

=== Environment ===
Virtual Env: $VIRTUAL_ENV
Python Path: $(python -c "import sys; print(sys.executable)")

=== Package Versions ===
$(pip list | grep -E "(docling|streamlit|tesserocr|pillow|opencv)" || echo "Packages not installed")

=== Tesseract ===
Version: $(tesseract --version 2>&1 | head -1 || echo "Not found")
Languages: $(tesseract --list-langs 2>&1 | tail -n +2 || echo "None")

=== Directory Structure ===
$(ls -la src/ 2>/dev/null || echo "src/ not found")
EOF

cat diagnostic.txt
```

### Common Next Steps

1. **Check the specific error message** against this guide
2. **Verify all prerequisites** are met
3. **Try installation in a clean environment**
4. **Check system-specific sections** above
5. **Review the Getting Started Tutorial** for basic usage
6. **Consult the Usage How-To Guides** for specific scenarios

### Emergency Reset

If all else fails, complete clean installation:

```bash
# Deactivate and remove virtual environment
deactivate
rm -rf venv

# Remove any cached files
rm -rf ~/.cache/pip
rm -rf ~/.cache/docling

# Start fresh
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Test basic functionality
python -c "import docling; print('Success!')"
```

This troubleshooting guide covers the most common installation issues. If you continue to experience problems after trying these solutions, the issue may be specific to your system configuration or a newer bug that needs investigation.