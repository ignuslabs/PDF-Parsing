# Development Workflow

*How-to guide for the development process and contribution workflow*

## Overview

Smart PDF Parser follows a **Test-Driven Development (TDD)** approach with strict code quality standards, automated testing, and continuous integration. This document outlines the complete development workflow from initial setup to production deployment.

## Quick Start for Contributors

### 1. Environment Setup

```bash
# Clone and enter repository
git clone <repository-url>
cd smart-pdf-parser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### 2. Generate Test Fixtures

**Critical:** Generate test fixtures before running any tests:

```bash
python generate_test_fixtures.py
```

This creates standardized PDF samples in `tests/fixtures/` for consistent testing across environments.

### 3. Run Quality Checks

```bash
# Complete quality check pipeline
black --check --diff src/ tests/ && \
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503 && \
mypy src/ --ignore-missing-imports --strict-optional
```

## Development Process

### TDD Workflow

Our development follows strict TDD principles:

1. **Write failing tests first**
2. **Write minimal code to pass**
3. **Refactor while keeping tests green**
4. **Ensure all quality gates pass**

#### Example TDD Cycle

```bash
# 1. Write failing test
pytest tests/test_new_feature.py::test_specific_case -v
# Should fail initially

# 2. Implement minimal functionality
# Edit src/module.py

# 3. Run test again
pytest tests/test_new_feature.py::test_specific_case -v
# Should pass

# 4. Refactor and run full suite
pytest tests/ -v -m "not slow and not ocr and not performance"

# 5. Quality checks
black src/ tests/
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
mypy src/ --ignore-missing-imports --strict-optional
```

### Git Workflow

We use **GitHub Flow** with protection rules:

```text
graph LR
    A[main] --> B[feature/branch]
    B --> C[Pull Request]
    C --> D[Code Review]
    D --> E[CI/CD Pipeline]
    E --> F[Merge to main]
```

#### Branch Naming Conventions

- `feature/parser-optimization` - New features
- `fix/ocr-memory-leak` - Bug fixes  
- `refactor/search-engine` - Code improvements
- `docs/api-documentation` - Documentation updates
- `test/integration-coverage` - Test improvements

#### Commit Message Format

```
type(scope): brief description

Detailed explanation if needed

Fixes #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

#### 1. Pre-PR Checklist

```bash
# Run complete test suite
pytest tests/ -v -m "not slow and not ocr and not performance"

# Quality checks pass
black --check --diff src/ tests/
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503  
mypy src/ --ignore-missing-imports --strict-optional

# Documentation updated (if applicable)
# Changelog entry added (if applicable)
```

#### 2. PR Template

```markdown
## Summary
Brief description of changes

## Changes Made
- [ ] Feature implementation
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Breaking changes noted

## Testing
- [ ] All tests pass locally
- [ ] New tests cover edge cases  
- [ ] Performance impact assessed

## Quality Gates
- [ ] Black formatting applied
- [ ] Flake8 linting passes
- [ ] MyPy type checking passes
- [ ] No security vulnerabilities
```

#### 3. Review Process

**Automatic checks** (must pass):
- All CI/CD pipelines green
- Code coverage ≥ 85%
- No security vulnerabilities
- Documentation builds successfully

**Manual review** (2 approvals required):
- Code quality and architecture
- Test coverage and quality
- Documentation completeness
- Breaking change assessment

## CI/CD Pipeline

### GitHub Actions Workflow

Our CI/CD runs on every push and PR with **four parallel jobs**:

#### Test Job (Matrix: Python 3.9, 3.10, 3.11)

```yaml
- name: Run fast tests
  run: |
    pytest tests/ -v --maxfail=1 --disable-warnings \
      -m "not slow and not ocr and not performance" \
      --cov=src --cov-report=xml --cov-report=term-missing

- name: Run slow tests (Python 3.11 only)
  run: |
    pytest tests/ -v --disable-warnings \
      -m "slow or ocr" --maxfail=3

- name: Run performance tests (main branch only)  
  run: |
    pytest tests/ -v --disable-warnings \
      -m "performance" --maxfail=2
```

#### Lint Job

```yaml
- name: Run black (check only)
  run: black --check --diff src/ tests/

- name: Run flake8  
  run: flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

- name: Run mypy
  run: mypy src/ --ignore-missing-imports --strict-optional
```

#### Security Job

```yaml
- name: Run safety check
  run: safety check --ignore 70612

- name: Run bandit security check
  run: |
    bandit -r src/ -f json -o bandit-report.json
    bandit -r src/ --severity-level medium
```

#### Integration Job (main branch only)

```yaml
- name: Run integration tests
  run: |
    pytest tests/ -v --disable-warnings \
      -m "integration" --maxfail=5 \
      --cov=src --cov-report=xml
```

### Pipeline Stages

```text
graph TD
    A[Code Push/PR] --> B{Fast Tests}
    B --> C{Linting}
    B --> D{Security}
    B --> E{Type Check}
    
    C --> F{All Checks Pass?}
    D --> F
    E --> F
    
    F -->|Yes| G[Slow Tests Python 3.11]
    F -->|No| H[❌ Pipeline Fails]
    
    G --> I{Main Branch?}
    I -->|Yes| J[Integration Tests]
    I -->|No| K[✅ PR Ready]
    
    J --> L[Performance Tests]
    L --> M[Deploy Ready]
```

### Quality Gates

All the following **must pass** before merge:

- ✅ **Test Coverage**: ≥85% line coverage
- ✅ **Fast Tests**: All unit tests pass  
- ✅ **Code Style**: Black formatting enforced
- ✅ **Linting**: Flake8 with max line length 100
- ✅ **Type Safety**: MyPy strict checking
- ✅ **Security**: No medium+ vulnerabilities
- ✅ **Performance**: No regressions on main branch

### Deployment Strategy

#### Staging Deployment (develop branch)
- Automatic deployment on successful CI
- Full integration test suite
- Manual acceptance testing

#### Production Deployment (main branch)  
- Manual approval required
- Blue-green deployment
- Automated rollback on failures
- Production smoke tests

## Local Development Tools

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### IDE Configuration

#### VS Code (.vscode/settings.json)
```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=100"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/",
        "-v",
        "-m", "not slow and not ocr and not performance"
    ]
}
```

### Performance Profiling

```bash
# Memory profiling
python -m memory_profiler src/core/parser.py

# Performance profiling  
python -c "
import cProfile
from src.core.parser import DoclingParser
cProfile.run('DoclingParser().parse_document(\"test.pdf\")')
"

# Line-by-line profiling
kernprof -l -v src/core/parser.py
```

## Configuration Management

### Environment Variables

```bash
# Development
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
export LOG_LEVEL=DEBUG
export RUN_SLOW_TESTS=true

# Testing  
export RUN_INTEGRATION_TESTS=true
export RUN_PERFORMANCE_TESTS=true

# CI/CD
export CODECOV_TOKEN=<token>
export SAFETY_API_KEY=<key>
```

### Configuration Files

- **pyproject.toml**: Package metadata, tool configuration
- **pytest.ini**: Test discovery and execution
- **.github/workflows/tests.yml**: CI/CD pipeline  
- **requirements.txt**: Production dependencies
- **requirements-dev.txt**: Development dependencies

## Troubleshooting Common Issues

### Test Fixtures Missing
```bash
# Error: FileNotFoundError: tests/fixtures/text_simple.pdf
python generate_test_fixtures.py
```

### OCR Dependencies Missing
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils

# macOS  
brew install tesseract poppler

# Windows
# Install from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Memory Issues in Tests
```bash
# Run with memory limit
pytest tests/ -v -m "not performance" --maxfail=1
```

### Type Checking Failures
```bash
# Check specific module
mypy src/core/parser.py --ignore-missing-imports

# Show error context
mypy src/ --ignore-missing-imports --show-error-context
```

## Release Process

### Version Bumping

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Create release PR

# After merge to main:
git tag v0.2.0
git push origin v0.2.0
```

### Release Checklist

- [ ] All tests pass on main branch
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Performance benchmarks stable
- [ ] Security scan passes
- [ ] Release notes prepared

---

*This workflow ensures code quality, test coverage, and reliable deployments while maintaining development velocity.*