# Smart PDF Parser Documentation

Welcome to the comprehensive documentation for Smart PDF Parser, a powerful document analysis tool built on IBM's Docling library. This documentation follows the [Diátaxis framework](https://diataxis.fr/) to provide both learning-oriented tutorials and problem-solving how-to guides.

## 📚 Documentation Structure

This documentation is organized into four main categories following the Diátaxis framework:

```{toctree}
:maxdepth: 2
:caption: 🎯 Tutorials (Learning-oriented)
:hidden:

getting-started
installation-tutorial
```

```{toctree}
:maxdepth: 2
:caption: 🛠️ How-To Guides (Problem-oriented)
:hidden:

installation-troubleshooting
usage-howto-guides
development-howto-guides
```

```{toctree}
:maxdepth: 2
:caption: 📚 Reference (Information-oriented)
:hidden:

system-architecture
development/workflow
development/code-quality
development/testing-strategy
development/implementation-history/agent-prompt
development/implementation-history/verification-ui-plan
design/field-extraction
```

```{toctree}
:maxdepth: 2
:caption: 🔧 Maintenance (Operations-oriented)
:hidden:

maintenance/monitoring
maintenance/troubleshooting
maintenance/updates
```

### 🎯 **Tutorials (Learning-oriented)**
*Step-by-step lessons that take you by the hand through a series of steps to complete a project*

- **[Getting Started Tutorial](getting-started.md)** - Your first experience with Smart PDF Parser
- **[Installation Tutorial](installation-tutorial.md)** - Complete setup from scratch

### 🛠️ **How-To Guides (Problem-oriented)**
*Step-by-step guides to help you solve a specific problem*

- **[Installation Troubleshooting](installation-troubleshooting.md)** - Fix setup issues and problems
- **[Usage How-To Guides](usage-howto-guides.md)** - Process different document types effectively
- **[Development How-To Guides](development-howto-guides.md)** - Extend and contribute to the project

### 📚 **Reference (Information-oriented)**
*Technical descriptions of the machinery and how to operate it*

- **[System Architecture](system-architecture.md)** - Core components and design patterns
- **[Development Workflow](development/workflow.md)** - Development processes and standards
- **[Code Quality Guidelines](development/code-quality.md)** - Standards and best practices
- **[Testing Strategy](development/testing-strategy.md)** - Test organization and methodologies
- **[Agent Prompt Implementation](development/implementation-history/agent-prompt.md)** - AI integration details
- **[Verification UI Plan](development/implementation-history/verification-ui-plan.md)** - UI architecture and design
- **[Field Extraction Design](design/field-extraction.md)** - Key-value pair extraction system architecture

### 🔧 **Maintenance (Operations-oriented)**
*Practical steps for maintaining and operating the system*

- **[System Monitoring](maintenance/monitoring.md)** - Performance and health monitoring
- **[Troubleshooting Guide](maintenance/troubleshooting.md)** - Operational issue resolution
- **[Updates and Migration](maintenance/updates.md)** - Version updates and data migration

---

## 🚀 Quick Start

**New to Smart PDF Parser?** Start with the **[Getting Started Tutorial](getting-started.md)**

**Having installation issues?** Check the **[Installation Troubleshooting Guide](installation-troubleshooting.md)**

**Want to process specific document types?** Browse the **[Usage How-To Guides](usage-howto-guides.md)**

---

## 📖 Tutorial Section

### [Getting Started Tutorial](getting-started.md)
*Complete beginner's guide to Smart PDF Parser*

Learn the essential workflow in this hands-on tutorial:
- ✅ Set up the application quickly
- ✅ Parse your first PDF document
- ✅ Search and verify content
- ✅ Export results in multiple formats
- ✅ Understand core concepts and best practices

**Duration**: 15-20 minutes  
**Prerequisites**: Python 3.9+, a sample PDF document

---

### [Installation Tutorial](installation-tutorial.md)
*Comprehensive setup guide for all platforms*

Master the complete installation process:
- 🔧 System requirements and compatibility matrix
- 🔧 Python environment setup (virtual environments)
- 🔧 Core dependency installation
- 🔧 Tesseract OCR installation (Windows, macOS, Linux)
- 🔧 Installation verification and testing
- 🔧 Performance optimization tips

**Duration**: 30-45 minutes  
**Prerequisites**: Basic command line familiarity

---

## 🛠️ How-To Guides Section

### [Installation Troubleshooting](installation-troubleshooting.md)
*Solve setup problems and installation issues*

**Quick Diagnostic Tools**:
- 🔍 Automated problem detection
- 🔍 System compatibility checks
- 🔍 Dependency verification scripts

**Common Issues Covered**:
- Python version problems
- Virtual environment issues
- Dependency installation failures
- **Tesseract Q&A Section** - Comprehensive OCR troubleshooting
- Memory and performance problems
- Platform-specific issues (Windows, macOS, Linux)

**When to use**: When installation doesn't work as expected

---

### [Usage How-To Guides](usage-howto-guides.md)
*Process different document types effectively*

#### Document Processing Guides:
- 📄 **Scientific Papers** - Handle formulas, citations, and academic structure
- 📄 **Business Reports** - Process financial data and corporate documents  
- 📄 **Legal Documents** - Maintain precision for contracts and legal content
- 📄 **Technical Manuals** - Preserve code blocks and procedural content
- 📄 **Forms and Applications** - Extract key-value pairs from structured documents
- 📄 **Scanned Documents** - Optimize OCR for image-based PDFs

#### Advanced Usage:
- 🌍 **Multi-language OCR Configuration** - Setup for different languages
- ⚡ **Performance Optimization** - Handle large documents and batch processing
- ✅ **Verification Workflows** - Interactive accuracy validation
- 📊 **Export Formats** - JSON, CSV, Markdown, HTML output options

**When to use**: When processing specific document types or using advanced features

---

### [Development How-To Guides](development-howto-guides.md)
*Extend and contribute to Smart PDF Parser*

#### Testing and Quality:
- 🧪 **Running Tests** - Test suite usage and categories
- 🧪 **Writing New Tests** - Unit, integration, and property-based testing
- 🧪 **Test-Driven Development** - Follow TDD practices

#### Extension Development:
- 🔧 **Adding New Parsers** - Support additional document formats
- 🔧 **Extending Search** - Implement new search algorithms  
- 🔧 **Contributing Guidelines** - Code standards and contribution process

**When to use**: When developing features, running tests, or contributing code

---

## 📋 Document Processing Workflow

Smart PDF Parser follows this core workflow:

```text
graph LR
    A[📄 Parse] --> B[🔍 Search]
    B --> C[✅ Verify]
    C --> D[📊 Export]
    
    A1[Extract Elements] --> A
    B1[Find Content] --> B
    C1[Validate Accuracy] --> C
    D1[Multiple Formats] --> D
```

1. **📄 Parse** - Extract elements and key-value pairs from PDFs using Docling
2. **🔍 Search** - Find content with exact, fuzzy, or semantic search
3. **✅ Verify** - Validate extraction accuracy with visual verification and KV highlighting
4. **📊 Export** - Output results in JSON, CSV, Markdown, or HTML

---

## 🎯 Choose Your Path

### I'm New Here
→ Start with **[Getting Started Tutorial](getting-started.md)**

### I Need to Install/Setup  
→ Follow **[Installation Tutorial](installation-tutorial.md)**  
→ If issues arise: **[Installation Troubleshooting](installation-troubleshooting.md)**

### I Want to Process Documents
→ Check **[Usage How-To Guides](usage-howto-guides.md)** for your document type

### I Want to Develop/Extend
→ Review **[Development How-To Guides](development-howto-guides.md)**

### I'm Having Problems
→ Search **[Installation Troubleshooting](installation-troubleshooting.md)** first  
→ Then check relevant how-to guide for your use case

---

## 💡 Key Features

**Smart PDF Parser** provides:

### 🧠 **Intelligent Document Processing**
- IBM Docling integration for accurate text extraction
- Multi-modal content recognition (text, tables, images, formulas)
- **Key-Value Extraction** for forms, applications, and structured documents
- Confidence scoring and quality assessment

### 🔍 **Advanced Search Capabilities** 
- Exact, fuzzy, and semantic search modes
- Element type and page filtering
- Context-aware result ranking

### ✅ **Interactive Verification System**
- Visual document overlay for accuracy checking
- **Key-Value Pair Visualization** with color-coded highlighting
- Correction workflow with state tracking
- Quality metrics and progress monitoring

### 📊 **Flexible Export Options**
- JSON for data processing
- CSV for spreadsheet analysis
- Markdown for documentation
- HTML for web publishing

### 🌍 **Multi-language Support**
- Tesseract OCR with 100+ languages
- Unicode text preservation
- Character encoding handling

---

## 📞 Getting Help

### Documentation Navigation
- **Browse by category** using the links above
- **Search for specific topics** within each guide
- **Follow cross-references** between related sections

### Troubleshooting Strategy
1. **Check error messages** - they often contain solution hints
2. **Review prerequisites** - ensure all requirements are met
3. **Try examples first** - test with simple documents
4. **Verify installation** - run diagnostic commands
5. **Consult troubleshooting guides** - comprehensive problem solving

### Best Practices
- **Start simple** - begin with text-based PDFs
- **Verify systematically** - check accuracy for important content
- **Save progress regularly** - backup verification work
- **Monitor performance** - watch resource usage for large documents

---

*This documentation is designed to get you productive with Smart PDF Parser quickly while providing comprehensive reference material for advanced usage. Whether you're extracting data from business reports, analyzing research papers, or building automated document processing workflows, these guides will help you achieve your goals efficiently.*