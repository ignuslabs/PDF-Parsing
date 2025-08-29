# Update and Upgrade Procedures

*How-to guide for maintaining, updating, and upgrading Smart PDF Parser components*

## Overview

This guide provides systematic procedures for keeping Smart PDF Parser up-to-date, managing dependency upgrades, and handling version migrations. It covers both routine maintenance and major version upgrades while ensuring system stability and data integrity.

## Update Categories

### Routine Maintenance Updates

**Security Updates** (Weekly):
- Python security patches
- Dependency vulnerability fixes
- System library updates

**Dependency Updates** (Monthly):
- Minor version updates for core packages
- Performance improvements
- Bug fixes

**Feature Updates** (Quarterly):
- New functionality
- UI improvements
- Performance optimizations

### Major Version Upgrades

**Python Version Upgrades** (Annually):
- Python runtime version changes
- Breaking compatibility changes
- Performance improvements

**Core Dependency Upgrades** (Semi-annually):
- Docling major version updates
- Streamlit framework upgrades
- Database schema changes

## Pre-Update Assessment

### System Health Check

```bash
#!/bin/bash
# Pre-update system assessment

echo "=== Pre-Update System Assessment ==="
echo "Date: $(date)"
echo

# 1. Current version inventory
echo "1. Current System State:"
echo "Python: $(python --version)"
echo "Smart PDF Parser: $(python -c "from src import __version__; print(__version__)" 2>/dev/null || echo "Unknown")"

echo -e "\nKey Dependencies:"
python -c "
deps = ['docling', 'streamlit', 'pandas', 'numpy', 'pillow']
for dep in deps:
    try:
        module = __import__(dep)
        version = getattr(module, '__version__', 'Unknown')
        print(f'  {dep}: {version}')
    except ImportError:
        print(f'  {dep}: Not installed')
"

# 2. System resources
echo -e "\n2. System Resources:"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"

# 3. Active processes
echo -e "\n3. Active Processes:"
ps aux | grep -E "(python|streamlit|smart-pdf)" | grep -v grep | wc -l | xargs echo "Running processes:"

# 4. Recent errors
echo -e "\n4. Recent Issues:"
if [ -f "logs/smart-pdf-parser.log" ]; then
    error_count=$(tail -n 1000 logs/smart-pdf-parser.log | grep -i error | wc -l)
    echo "Recent errors (last 1000 log entries): $error_count"
    
    if [ "$error_count" -gt 10 ]; then
        echo "⚠️  High error rate detected - review logs before updating"
    fi
else
    echo "No log file found"
fi

# 5. Data backup check
echo -e "\n5. Data Backup Status:"
if [ -d "backups" ]; then
    latest_backup=$(ls -t backups/ | head -1 2>/dev/null)
    if [ -n "$latest_backup" ]; then
        echo "Latest backup: $latest_backup"
    else
        echo "⚠️  No backups found"
    fi
else
    echo "⚠️  Backup directory not found"
fi

echo -e "\n=== Assessment Complete ==="
```

### Dependency Vulnerability Scan

```bash
#!/bin/bash
# Check for security vulnerabilities

echo "=== Security Vulnerability Scan ==="

# Install/update safety tool
pip install --upgrade safety

# Check for known vulnerabilities
echo "Checking for known vulnerabilities..."
safety check --json > vulnerability_report.json 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ No known vulnerabilities found"
else
    echo "⚠️  Vulnerabilities detected - see vulnerability_report.json"
    
    # Show summary
    python -c "
import json
try:
    with open('vulnerability_report.json', 'r') as f:
        data = json.load(f)
        if data:
            print(f'Found {len(data)} vulnerabilities:')
            for vuln in data[:5]:  # Show first 5
                print(f'  - {vuln.get(\"package\", \"unknown\")}: {vuln.get(\"vulnerability\", \"N/A\")}')
        else:
            print('No vulnerabilities in report')
except Exception as e:
    print(f'Error reading vulnerability report: {e}')
"
fi

# Check outdated packages
echo -e "\nChecking for outdated packages..."
pip list --outdated --format=json > outdated_packages.json

python -c "
import json
try:
    with open('outdated_packages.json', 'r') as f:
        packages = json.load(f)
        if packages:
            print(f'Found {len(packages)} outdated packages:')
            for pkg in packages[:10]:  # Show first 10
                print(f'  {pkg[\"name\"]}: {pkg[\"version\"]} -> {pkg[\"latest_version\"]}')
        else:
            print('All packages are up to date')
except Exception as e:
    print(f'Error reading outdated packages: {e}')
"
```

### Compatibility Testing

```python
"""
Test suite for update compatibility.
"""

import subprocess
import sys
from typing import Dict, List, Tuple
import importlib

class CompatibilityTester:
    """Test compatibility before and after updates."""
    
    def __init__(self):
        self.test_results = []
        self.critical_imports = [
            'docling', 'streamlit', 'pandas', 'numpy', 
            'pillow', 'opencv-python', 'fuzzywuzzy'
        ]
    
    def test_import_compatibility(self) -> Dict[str, bool]:
        """Test if all critical packages can be imported."""
        
        results = {}
        
        for package in self.critical_imports:
            try:
                # Handle packages with different import names
                import_name = {
                    'opencv-python': 'cv2',
                    'pillow': 'PIL'
                }.get(package, package)
                
                module = importlib.import_module(import_name)
                results[package] = True
                
                # Check for version compatibility
                version = getattr(module, '__version__', 'Unknown')
                print(f"✓ {package}: {version}")
                
            except ImportError as e:
                results[package] = False
                print(f"✗ {package}: Import failed - {e}")
            except Exception as e:
                results[package] = False
                print(f"✗ {package}: Error - {e}")
        
        return results
    
    def test_core_functionality(self) -> bool:
        """Test core application functionality."""
        
        try:
            # Test parser initialization
            from src.core.parser import DoclingParser
            parser = DoclingParser()
            print("✓ Parser initialization")
            
            # Test search engine
            from src.core.search import SmartSearchEngine
            from src.core.models import DocumentElement
            
            # Create dummy element
            dummy_element = DocumentElement(
                text="Test document",
                element_type="paragraph",
                bbox=(0, 0, 100, 20),
                page_num=1
            )
            
            engine = SmartSearchEngine([dummy_element])
            results = engine.search("test")
            print("✓ Search engine functionality")
            
            # Test verification interface
            from src.verification.interface import VerificationInterface
            interface = VerificationInterface()
            print("✓ Verification interface")
            
            return True
            
        except Exception as e:
            print(f"✗ Core functionality test failed: {e}")
            return False
    
    def test_file_processing(self, test_file: str = "tests/fixtures/text_simple.pdf") -> bool:
        """Test actual file processing capability."""
        
        try:
            from pathlib import Path
            from src.core.parser import DoclingParser
            
            # Check if test file exists
            if not Path(test_file).exists():
                print(f"⚠️  Test file not found: {test_file}")
                return True  # Not a failure, just can't test
            
            # Test parsing
            parser = DoclingParser(enable_ocr=False)  # Fast test
            elements = parser.parse_document(test_file)
            
            if elements and len(elements) > 0:
                print(f"✓ File processing: {len(elements)} elements extracted")
                return True
            else:
                print("⚠️  File processing returned no elements")
                return False
                
        except Exception as e:
            print(f"✗ File processing test failed: {e}")
            return False
    
    def run_compatibility_tests(self) -> Tuple[bool, Dict]:
        """Run all compatibility tests."""
        
        print("=== Compatibility Test Suite ===")
        
        # Import tests
        import_results = self.test_import_compatibility()
        imports_ok = all(import_results.values())
        
        print(f"\nImport Tests: {'✓ PASS' if imports_ok else '✗ FAIL'}")
        
        # Core functionality tests
        core_ok = self.test_core_functionality()
        print(f"Core Functionality: {'✓ PASS' if core_ok else '✗ FAIL'}")
        
        # File processing tests
        file_ok = self.test_file_processing()
        print(f"File Processing: {'✓ PASS' if file_ok else '✗ FAIL'}")
        
        overall_result = imports_ok and core_ok and file_ok
        
        return overall_result, {
            'imports': import_results,
            'core_functionality': core_ok,
            'file_processing': file_ok
        }

# Usage
def run_pre_update_compatibility_test():
    """Run compatibility test before updates."""
    
    tester = CompatibilityTester()
    success, results = tester.run_compatibility_tests()
    
    if success:
        print("\n✅ All compatibility tests passed - safe to proceed with update")
        return True
    else:
        print("\n❌ Compatibility issues detected - review before updating")
        return False

# Run pre-update test
if __name__ == "__main__":
    run_pre_update_compatibility_test()
```

## Backup Procedures

### Data Backup Strategy

```bash
#!/bin/bash
# Comprehensive backup script

BACKUP_DIR="backups/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BACKUP_DIR"

echo "=== Creating System Backup ==="
echo "Backup location: $BACKUP_DIR"

# 1. Application code backup
echo "1. Backing up application code..."
tar -czf "$BACKUP_DIR/application_code.tar.gz" \
    src/ \
    tests/ \
    docs/ \
    requirements*.txt \
    pyproject.toml \
    pytest.ini \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    2>/dev/null

# 2. Configuration backup
echo "2. Backing up configuration..."
cp -r config/ "$BACKUP_DIR/" 2>/dev/null || echo "No config directory found"

# 3. Database backup
echo "3. Backing up databases..."
if [ -f "app_state.db" ]; then
    cp app_state.db "$BACKUP_DIR/"
fi

if [ -d "data/" ]; then
    tar -czf "$BACKUP_DIR/data.tar.gz" data/ 2>/dev/null
fi

# 4. Logs backup
echo "4. Backing up logs..."
if [ -d "logs/" ]; then
    tar -czf "$BACKUP_DIR/logs.tar.gz" logs/ 2>/dev/null
fi

# 5. Virtual environment requirements
echo "5. Saving environment state..."
pip freeze > "$BACKUP_DIR/pip_freeze.txt"
python --version > "$BACKUP_DIR/python_version.txt" 2>&1
pip list --format=json > "$BACKUP_DIR/pip_list.json"

# 6. System information
echo "6. Collecting system information..."
cat > "$BACKUP_DIR/system_info.txt" << EOF
Backup Date: $(date)
Hostname: $(hostname)
OS: $(uname -a)
Python Path: $(which python)
Disk Usage: $(df -h /)
Memory: $(free -h)
EOF

# 7. Create restore script
cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Restore script - run from project root directory

echo "=== Restoring Smart PDF Parser Backup ==="
read -p "This will overwrite current installation. Continue? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Stop running services
    pkill -f "streamlit" 2>/dev/null || true
    
    # Restore application code
    tar -xzf application_code.tar.gz
    
    # Restore configuration
    cp -r config/ ./ 2>/dev/null || true
    
    # Restore databases
    cp *.db ./ 2>/dev/null || true
    
    # Restore data
    tar -xzf data.tar.gz 2>/dev/null || true
    
    # Restore logs
    tar -xzf logs.tar.gz 2>/dev/null || true
    
    echo "✓ Backup restored"
    echo "✓ Reinstall dependencies: pip install -r requirements.txt"
else
    echo "Restore cancelled"
fi
EOF

chmod +x "$BACKUP_DIR/restore.sh"

echo "✅ Backup complete: $BACKUP_DIR"
echo "💾 Backup size: $(du -sh "$BACKUP_DIR" | cut -f1)"
```

### Configuration Backup

```python
"""
Configuration and state backup utilities.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class ConfigurationBackup:
    """Backup and restore application configuration."""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_streamlit_config(self) -> Dict[str, Any]:
        """Backup Streamlit configuration."""
        
        config_data = {}
        
        # Backup .streamlit/config.toml if it exists
        config_file = Path(".streamlit/config.toml")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data['streamlit_config'] = f.read()
        
        # Backup secrets if they exist
        secrets_file = Path(".streamlit/secrets.toml")
        if secrets_file.exists():
            with open(secrets_file, 'r') as f:
                config_data['streamlit_secrets'] = f.read()
        
        return config_data
    
    def backup_database_schema(self, db_path: str = "app_state.db") -> Dict[str, Any]:
        """Backup database schema and critical data."""
        
        if not Path(db_path).exists():
            return {}
        
        backup_data = {}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
            schemas = cursor.fetchall()
            backup_data['schemas'] = [schema[0] for schema in schemas if schema[0]]
            
            # Backup critical tables (avoid large session data)
            critical_tables = ['user_preferences', 'system_config', 'verification_templates']
            
            for table in critical_tables:
                try:
                    cursor.execute(f"SELECT * FROM {table}")
                    backup_data[f"table_{table}"] = cursor.fetchall()
                except sqlite3.Error:
                    # Table doesn't exist, skip
                    pass
            
            conn.close()
            
        except Exception as e:
            backup_data['error'] = str(e)
        
        return backup_data
    
    def backup_user_preferences(self) -> Dict[str, Any]:
        """Backup user preferences and customizations."""
        
        preferences = {}
        
        # Default parser settings
        preferences['default_parser_config'] = {
            'enable_ocr': False,
            'enable_tables': True,
            'generate_page_images': False,
            'ocr_language': 'eng'
        }
        
        # UI preferences
        preferences['ui_config'] = {
            'theme': 'light',
            'sidebar_expanded': True,
            'auto_save': True
        }
        
        # Load from file if exists
        prefs_file = Path("config/user_preferences.json")
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    saved_prefs = json.load(f)
                    preferences.update(saved_prefs)
            except Exception as e:
                preferences['load_error'] = str(e)
        
        return preferences
    
    def create_full_backup(self) -> str:
        """Create complete configuration backup."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"config_backup_{timestamp}.json"
        
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'streamlit_config': self.backup_streamlit_config(),
            'database_schema': self.backup_database_schema(),
            'user_preferences': self.backup_user_preferences()
        }
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return str(backup_file)
    
    def restore_configuration(self, backup_file: str) -> bool:
        """Restore configuration from backup."""
        
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Restore Streamlit config
            streamlit_config = backup_data.get('streamlit_config', {})
            if 'streamlit_config' in streamlit_config:
                Path(".streamlit").mkdir(exist_ok=True)
                with open(".streamlit/config.toml", 'w') as f:
                    f.write(streamlit_config['streamlit_config'])
            
            # Restore user preferences  
            user_prefs = backup_data.get('user_preferences', {})
            if user_prefs:
                Path("config").mkdir(exist_ok=True)
                with open("config/user_preferences.json", 'w') as f:
                    json.dump(user_prefs, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Configuration restore failed: {e}")
            return False

# Usage
backup_manager = ConfigurationBackup()
backup_file = backup_manager.create_full_backup()
print(f"Configuration backup created: {backup_file}")
```

## Update Execution

### Safe Update Process

```bash
#!/bin/bash
# Safe update execution script

set -e  # Exit on any error

echo "=== Smart PDF Parser Update Process ==="

# Check if running as correct user
if [ "$EUID" -eq 0 ]; then
    echo "❌ Do not run updates as root"
    exit 1
fi

# 1. Pre-update checks
echo "1. Running pre-update checks..."

# Check if application is running
if pgrep -f "streamlit" > /dev/null; then
    echo "⚠️  Application is currently running"
    read -p "Stop application and continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "streamlit" || true
        sleep 2
    else
        echo "Update cancelled"
        exit 1
    fi
fi

# 2. Create backup
echo "2. Creating backup..."
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/pre_update_$BACKUP_TIMESTAMP"

mkdir -p "$BACKUP_DIR"

# Quick backup of essential files
cp requirements.txt "$BACKUP_DIR/" 2>/dev/null || true
cp -r src/ "$BACKUP_DIR/" 2>/dev/null || true
cp *.db "$BACKUP_DIR/" 2>/dev/null || true
pip freeze > "$BACKUP_DIR/pip_freeze_before.txt"

echo "✓ Backup created: $BACKUP_DIR"

# 3. Update dependencies
echo "3. Updating dependencies..."

# Create virtual environment backup
if [ -d "venv" ]; then
    echo "Backing up virtual environment..."
    cp -r venv "$BACKUP_DIR/venv_backup" 2>/dev/null || echo "Virtual env backup failed (continuing)"
fi

# Update pip first
python -m pip install --upgrade pip

# Install updates
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt --upgrade
fi

if [ -f "requirements-dev.txt" ]; then
    echo "Installing dev dependencies..."
    pip install -r requirements-dev.txt --upgrade
fi

# 4. Post-update verification
echo "4. Running post-update verification..."

# Test imports
python -c "
import sys
sys.path.insert(0, 'src')

try:
    from core.parser import DoclingParser
    from core.search import SmartSearchEngine
    print('✓ Core imports successful')
except Exception as e:
    print(f'✗ Import test failed: {e}')
    sys.exit(1)
"

# 5. Save post-update state
pip freeze > "$BACKUP_DIR/pip_freeze_after.txt"

echo "5. Generating update report..."
cat > "$BACKUP_DIR/update_report.txt" << EOF
Update Report
=============
Date: $(date)
Backup Directory: $BACKUP_DIR

Package Changes:
$(diff "$BACKUP_DIR/pip_freeze_before.txt" "$BACKUP_DIR/pip_freeze_after.txt" | head -20)

System State:
- Python: $(python --version)
- Disk Space: $(df -h / | tail -1 | awk '{print $4}' | sed 's/Available//')
- Memory: $(free -h | grep Mem | awk '{print $7}')

Next Steps:
1. Test application functionality
2. Run compatibility tests  
3. Monitor for issues in first 24 hours
EOF

echo "✅ Update completed successfully"
echo "📄 Update report: $BACKUP_DIR/update_report.txt"
echo "🔄 Test the application: streamlit run src/ui/app.py"

# Optional: restart application
read -p "Start application now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    nohup streamlit run src/ui/app.py &
    echo "Application started in background"
fi
```

### Rollback Procedures

```bash
#!/bin/bash
# Rollback to previous version

echo "=== Smart PDF Parser Rollback Process ==="

# Find latest backup
LATEST_BACKUP=$(ls -t backups/ | grep "pre_update" | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ No update backup found"
    exit 1
fi

BACKUP_PATH="backups/$LATEST_BACKUP"
echo "Found backup: $BACKUP_PATH"

# Confirm rollback
read -p "Rollback to backup from $LATEST_BACKUP? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled"
    exit 0
fi

# Stop application
echo "1. Stopping application..."
pkill -f "streamlit" 2>/dev/null || true
sleep 2

# Create current state backup before rollback
ROLLBACK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CURRENT_BACKUP="backups/before_rollback_$ROLLBACK_TIMESTAMP"
mkdir -p "$CURRENT_BACKUP"

echo "2. Backing up current state..."
cp -r src/ "$CURRENT_BACKUP/" 2>/dev/null || true
pip freeze > "$CURRENT_BACKUP/pip_freeze.txt"

# Restore from backup
echo "3. Restoring from backup..."

if [ -f "$BACKUP_PATH/requirements.txt" ]; then
    cp "$BACKUP_PATH/requirements.txt" ./
fi

if [ -d "$BACKUP_PATH/src" ]; then
    rm -rf src/
    cp -r "$BACKUP_PATH/src" ./
fi

# Restore database
if [ -f "$BACKUP_PATH/app_state.db" ]; then
    cp "$BACKUP_PATH/app_state.db" ./
fi

# Restore dependencies
if [ -f "$BACKUP_PATH/pip_freeze_before.txt" ]; then
    echo "4. Restoring dependencies..."
    pip install -r "$BACKUP_PATH/pip_freeze_before.txt" --force-reinstall --no-deps
fi

# Verify rollback
echo "5. Verifying rollback..."
python -c "
try:
    from src.core.parser import DoclingParser
    print('✓ Rollback verification successful')
except Exception as e:
    print(f'✗ Rollback verification failed: {e}')
"

echo "✅ Rollback completed"
echo "🔄 Restart application: streamlit run src/ui/app.py"

# Create rollback report
cat > "rollback_report_$ROLLBACK_TIMESTAMP.txt" << EOF
Rollback Report
===============
Date: $(date)
Rolled back to: $LATEST_BACKUP
Current state backed up to: $CURRENT_BACKUP

Actions taken:
1. Stopped running application
2. Backed up current state  
3. Restored code from backup
4. Restored dependencies
5. Verified functionality

Next steps:
1. Test application functionality
2. Investigate why rollback was needed
3. Plan corrective update if necessary
EOF

echo "📄 Rollback report: rollback_report_$ROLLBACK_TIMESTAMP.txt"
```

## Dependency Management

### Upgrade Strategy

```python
"""
Systematic dependency upgrade management.
"""

import subprocess
import json
from typing import List, Dict, Tuple
from packaging import version
import requests

class DependencyUpgradeManager:
    """Manage dependency upgrades with safety checks."""
    
    def __init__(self):
        self.critical_packages = [
            'docling', 'streamlit', 'pandas', 'numpy', 
            'pillow', 'opencv-python', 'fuzzywuzzy'
        ]
        self.upgrade_order = [
            # Upgrade order matters for compatibility
            ['numpy'],  # Foundation packages first
            ['pandas'], 
            ['pillow'],
            ['opencv-python'],
            ['fuzzywuzzy'],
            ['streamlit'],
            ['docling']  # Core application dependency last
        ]
    
    def get_outdated_packages(self) -> List[Dict]:
        """Get list of outdated packages."""
        
        try:
            result = subprocess.run(['pip', 'list', '--outdated', '--format=json'], 
                                  capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error getting outdated packages: {e}")
            return []
    
    def check_package_compatibility(self, package: str, target_version: str) -> Dict:
        """Check if package version is compatible with requirements."""
        
        compatibility = {
            'compatible': True,
            'issues': [],
            'recommendations': []
        }
        
        # Version-specific compatibility checks
        if package == 'streamlit':
            if version.parse(target_version) >= version.parse('1.30.0'):
                compatibility['recommendations'].append(
                    "Streamlit 1.30+ may have UI changes - test thoroughly"
                )
        
        elif package == 'docling':
            if version.parse(target_version) >= version.parse('2.0.0'):
                compatibility['issues'].append(
                    "Docling 2.0+ has breaking API changes"
                )
                compatibility['compatible'] = False
        
        elif package == 'pandas':
            if version.parse(target_version) >= version.parse('2.1.0'):
                compatibility['recommendations'].append(
                    "Pandas 2.1+ has performance improvements but check for dtype changes"
                )
        
        return compatibility
    
    def upgrade_packages_safely(self, dry_run: bool = True) -> Dict:
        """Perform safe package upgrades."""
        
        outdated = self.get_outdated_packages()
        upgrade_plan = []
        
        # Group packages by upgrade order
        for order_group in self.upgrade_order:
            group_upgrades = []
            
            for package_info in outdated:
                package_name = package_info['name']
                current_version = package_info['version']
                latest_version = package_info['latest_version']
                
                if package_name in order_group:
                    # Check compatibility
                    compat = self.check_package_compatibility(package_name, latest_version)
                    
                    if compat['compatible']:
                        group_upgrades.append({
                            'package': package_name,
                            'current': current_version,
                            'target': latest_version,
                            'compatibility': compat
                        })
                    else:
                        print(f"⚠️  Skipping {package_name} due to compatibility issues:")
                        for issue in compat['issues']:
                            print(f"   - {issue}")
            
            if group_upgrades:
                upgrade_plan.append(group_upgrades)
        
        # Execute upgrades if not dry run
        results = {'planned': upgrade_plan, 'executed': []}
        
        if not dry_run:
            for group in upgrade_plan:
                group_results = []
                
                for upgrade in group:
                    package = upgrade['package']
                    target = upgrade['target']
                    
                    print(f"Upgrading {package} to {target}...")
                    
                    try:
                        subprocess.run(['pip', 'install', f'{package}=={target}'], 
                                     check=True, capture_output=True)
                        
                        # Test import after upgrade
                        subprocess.run(['python', '-c', f'import {package}'], 
                                     check=True, capture_output=True)
                        
                        group_results.append({
                            'package': package,
                            'success': True,
                            'version': target
                        })
                        print(f"✓ {package} upgraded successfully")
                        
                    except subprocess.CalledProcessError as e:
                        group_results.append({
                            'package': package,
                            'success': False,
                            'error': str(e)
                        })
                        print(f"✗ {package} upgrade failed: {e}")
                        
                        # Rollback group on failure
                        print(f"Rolling back group due to {package} failure...")
                        self.rollback_group_upgrades(group[:group.index(upgrade)])
                        break
                
                results['executed'].append(group_results)
        
        return results
    
    def rollback_group_upgrades(self, upgraded_packages: List[Dict]):
        """Rollback a group of package upgrades."""
        
        for upgrade in reversed(upgraded_packages):
            package = upgrade['package']
            original_version = upgrade['current']
            
            try:
                subprocess.run(['pip', 'install', f'{package}=={original_version}'], 
                             check=True, capture_output=True)
                print(f"✓ Rolled back {package} to {original_version}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to rollback {package}: {e}")

# Usage
def perform_safe_upgrade():
    """Perform safe dependency upgrade."""
    
    manager = DependencyUpgradeManager()
    
    print("=== Dependency Upgrade Analysis ===")
    
    # Dry run first
    plan = manager.upgrade_packages_safely(dry_run=True)
    
    if not plan['planned']:
        print("✓ All packages are up to date")
        return
    
    # Show upgrade plan
    print("\nUpgrade Plan:")
    for i, group in enumerate(plan['planned'], 1):
        print(f"\nGroup {i}:")
        for upgrade in group:
            print(f"  {upgrade['package']}: {upgrade['current']} → {upgrade['target']}")
            for rec in upgrade['compatibility']['recommendations']:
                print(f"    ℹ️  {rec}")
    
    # Confirm execution
    response = input("\nExecute upgrade plan? (y/N): ")
    if response.lower() == 'y':
        results = manager.upgrade_packages_safely(dry_run=False)
        
        # Report results
        print("\n=== Upgrade Results ===")
        for group_results in results['executed']:
            for result in group_results:
                status = "✓" if result['success'] else "✗"
                print(f"{status} {result['package']}: {result.get('version', 'Failed')}")

if __name__ == "__main__":
    perform_safe_upgrade()
```

## Version Migration

### Major Version Upgrade Process

```python
"""
Handle major version upgrades with data migration.
"""

import shutil
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class VersionMigration:
    """Handle version migrations and data compatibility."""
    
    def __init__(self):
        self.current_version = self.get_current_version()
        self.migration_scripts = {
            '0.1.0': self.migrate_to_v0_1_0,
            '0.2.0': self.migrate_to_v0_2_0,
            '1.0.0': self.migrate_to_v1_0_0
        }
    
    def get_current_version(self) -> str:
        """Get current application version."""
        
        try:
            # Try to get from package
            from src import __version__
            return __version__
        except ImportError:
            # Fall back to pyproject.toml
            try:
                import toml
                with open('pyproject.toml', 'r') as f:
                    config = toml.load(f)
                    return config['project']['version']
            except:
                return '0.0.0'
    
    def get_data_version(self) -> str:
        """Get version of existing data."""
        
        version_file = Path('data_version.txt')
        if version_file.exists():
            return version_file.read_text().strip()
        
        # Check database for version info
        if Path('app_state.db').exists():
            try:
                conn = sqlite3.connect('app_state.db')
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM metadata WHERE key = 'version'")
                result = cursor.fetchone()
                conn.close()
                if result:
                    return result[0]
            except:
                pass
        
        return '0.0.0'
    
    def needs_migration(self, target_version: str) -> bool:
        """Check if migration is needed."""
        
        data_version = self.get_data_version()
        
        from packaging import version
        return version.parse(data_version) < version.parse(target_version)
    
    def migrate_to_v0_1_0(self) -> Dict[str, Any]:
        """Migration to version 0.1.0."""
        
        migration_log = {
            'version': '0.1.0',
            'date': datetime.now().isoformat(),
            'actions': [],
            'success': True
        }
        
        try:
            # Create new directory structure
            Path('logs').mkdir(exist_ok=True)
            Path('backups').mkdir(exist_ok=True)
            Path('config').mkdir(exist_ok=True)
            
            migration_log['actions'].append('Created directory structure')
            
            # Initialize database schema
            self.initialize_database_v0_1_0()
            migration_log['actions'].append('Initialized database schema')
            
        except Exception as e:
            migration_log['success'] = False
            migration_log['error'] = str(e)
        
        return migration_log
    
    def migrate_to_v0_2_0(self) -> Dict[str, Any]:
        """Migration to version 0.2.0."""
        
        migration_log = {
            'version': '0.2.0',
            'date': datetime.now().isoformat(),
            'actions': [],
            'success': True
        }
        
        try:
            # Update database schema
            conn = sqlite3.connect('app_state.db')
            cursor = conn.cursor()
            
            # Add new columns if they don't exist
            try:
                cursor.execute("""
                    ALTER TABLE app_state 
                    ADD COLUMN version TEXT DEFAULT '0.2.0'
                """)
                migration_log['actions'].append('Added version column to app_state')
            except sqlite3.Error:
                # Column already exists
                pass
            
            # Migrate session data format
            cursor.execute("SELECT session_id, key, value FROM app_state")
            rows = cursor.fetchall()
            
            for session_id, key, value in rows:
                if key == 'parsed_elements':
                    # Convert old format to new format
                    try:
                        import pickle
                        elements = pickle.loads(value)
                        
                        # Update element structure if needed
                        updated_elements = []
                        for elem in elements:
                            if isinstance(elem, dict) and 'bbox' in elem:
                                # Ensure bbox has 4 coordinates
                                bbox = elem['bbox']
                                if len(bbox) == 2:  # Old format had x,y only
                                    elem['bbox'] = (bbox[0], bbox[1], bbox[0]+100, bbox[1]+20)
                                updated_elements.append(elem)
                        
                        # Save updated elements
                        updated_value = pickle.dumps(updated_elements)
                        cursor.execute("""
                            UPDATE app_state 
                            SET value = ?, version = '0.2.0'
                            WHERE session_id = ? AND key = ?
                        """, (updated_value, session_id, key))
                        
                        migration_log['actions'].append(f'Updated parsed elements for session {session_id}')
                        
                    except Exception as e:
                        migration_log['actions'].append(f'Failed to migrate session {session_id}: {e}')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            migration_log['success'] = False
            migration_log['error'] = str(e)
        
        return migration_log
    
    def migrate_to_v1_0_0(self) -> Dict[str, Any]:
        """Migration to version 1.0.0."""
        
        migration_log = {
            'version': '1.0.0',
            'date': datetime.now().isoformat(),
            'actions': [],
            'success': True
        }
        
        try:
            # Major schema changes for 1.0.0
            self.backup_database_for_migration('1.0.0')
            migration_log['actions'].append('Created database backup')
            
            # Recreate database with new schema
            self.initialize_database_v1_0_0()
            migration_log['actions'].append('Initialized v1.0.0 database schema')
            
            # Migrate user preferences to new format
            old_config = Path('config/user_preferences.json')
            if old_config.exists():
                with open(old_config, 'r') as f:
                    old_prefs = json.load(f)
                
                new_prefs = self.convert_preferences_v1_0_0(old_prefs)
                
                with open('config/user_preferences_v1.json', 'w') as f:
                    json.dump(new_prefs, f, indent=2)
                
                migration_log['actions'].append('Migrated user preferences')
            
        except Exception as e:
            migration_log['success'] = False
            migration_log['error'] = str(e)
        
        return migration_log
    
    def initialize_database_v0_1_0(self):
        """Initialize database schema for v0.1.0."""
        
        conn = sqlite3.connect('app_state.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_state (
                session_id TEXT,
                key TEXT,
                value BLOB,
                timestamp REAL,
                PRIMARY KEY (session_id, key)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at REAL
            )
        """)
        
        # Set version
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES ('version', '0.1.0', ?)
        """, (time.time(),))
        
        conn.commit()
        conn.close()
    
    def initialize_database_v1_0_0(self):
        """Initialize database schema for v1.0.0."""
        
        # Backup existing database
        if Path('app_state.db').exists():
            shutil.copy('app_state.db', 'app_state_backup_v0.db')
        
        conn = sqlite3.connect('app_state.db')
        cursor = conn.cursor()
        
        # New schema with improved structure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL,
                last_accessed REAL,
                user_agent TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_cache (
                document_hash TEXT PRIMARY KEY,
                file_path TEXT,
                parsed_elements BLOB,
                created_at REAL,
                file_size INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                session_id TEXT,
                preference_key TEXT,
                preference_value TEXT,
                updated_at REAL,
                PRIMARY KEY (session_id, preference_key)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def perform_migration(self, target_version: str) -> Dict[str, Any]:
        """Perform migration to target version."""
        
        if not self.needs_migration(target_version):
            return {
                'needed': False,
                'message': f'Already at version {target_version} or higher'
            }
        
        migration_results = {
            'needed': True,
            'target_version': target_version,
            'migrations_performed': [],
            'overall_success': True
        }
        
        # Perform migrations in sequence
        from packaging import version
        
        for migration_version in sorted(self.migration_scripts.keys(), key=version.parse):
            if version.parse(migration_version) <= version.parse(target_version):
                data_version = self.get_data_version()
                
                if version.parse(data_version) < version.parse(migration_version):
                    print(f"Performing migration to {migration_version}...")
                    
                    result = self.migration_scripts[migration_version]()
                    migration_results['migrations_performed'].append(result)
                    
                    if result['success']:
                        # Update data version
                        with open('data_version.txt', 'w') as f:
                            f.write(migration_version)
                        print(f"✓ Migration to {migration_version} completed")
                    else:
                        migration_results['overall_success'] = False
                        print(f"✗ Migration to {migration_version} failed: {result.get('error', 'Unknown error')}")
                        break
        
        return migration_results

# Usage
def run_version_migration(target_version: str):
    """Run version migration process."""
    
    migrator = VersionMigration()
    
    print(f"=== Version Migration to {target_version} ===")
    print(f"Current version: {migrator.current_version}")
    print(f"Data version: {migrator.get_data_version()}")
    
    if migrator.needs_migration(target_version):
        result = migrator.perform_migration(target_version)
        
        if result['overall_success']:
            print("✅ All migrations completed successfully")
        else:
            print("❌ Migration failed - check logs and rollback if necessary")
            
        return result
    else:
        print("✓ No migration needed")
        return {'needed': False}
```

---

*This update and upgrade guide ensures safe, systematic maintenance of Smart PDF Parser while preserving data integrity and system stability.*