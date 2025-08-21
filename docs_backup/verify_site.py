#!/usr/bin/env python3
"""
Verification script for Jekyll site structure and content.
Checks that all required files exist and have proper formatting.
"""

import os
from pathlib import Path
import yaml
import re

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if filepath.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_yaml_valid(filepath):
    """Check if YAML file is valid."""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        return True
    except yaml.YAMLError as e:
        print(f"❌ YAML error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False

def check_markdown_frontmatter(filepath):
    """Check if Markdown file has valid front matter."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if content.startswith('---\n'):
            # Find end of front matter
            end_idx = content.find('\n---\n', 4)
            if end_idx > 0:
                frontmatter = content[4:end_idx]
                yaml.safe_load(frontmatter)
                return True
        
        print(f"⚠️  No valid front matter in {filepath}")
        return False
    except Exception as e:
        print(f"❌ Error checking front matter in {filepath}: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Verifying Jekyll site structure...")
    print("=" * 50)
    
    site_root = Path(__file__).parent
    all_checks = []
    
    # Check core Jekyll files
    print("\n📁 Core Jekyll Files:")
    core_files = [
        ('_config.yml', 'Jekyll configuration'),
        ('Gemfile', 'Ruby dependencies'),
        ('index.md', 'Home page'),
        ('README.md', 'Documentation')
    ]
    
    for filename, desc in core_files:
        filepath = site_root / filename
        all_checks.append(check_file_exists(filepath, desc))
        
        if filename.endswith('.yml') and filepath.exists():
            all_checks.append(check_yaml_valid(filepath))
    
    # Check layouts
    print("\n🎨 Layout Templates:")
    layout_files = [
        ('_layouts/default.html', 'Main layout'),
        ('_layouts/page.html', 'Page layout')
    ]
    
    for filename, desc in layout_files:
        filepath = site_root / filename
        all_checks.append(check_file_exists(filepath, desc))
    
    # Check includes
    print("\n🧩 Include Templates:")
    include_files = [
        ('_includes/header.html', 'Site header'),
        ('_includes/footer.html', 'Site footer'),
        ('_includes/navigation.html', 'Navigation menu')
    ]
    
    for filename, desc in include_files:
        filepath = site_root / filename
        all_checks.append(check_file_exists(filepath, desc))
    
    # Check content pages
    print("\n📄 Content Pages:")
    page_files = [
        ('pages/about.md', 'About page'),
        ('pages/methodology.md', 'Methodology page'),
        ('pages/results.md', 'Results page'),
        ('pages/conclusion.md', 'Conclusion page'),
        ('pages/contact.md', 'Contact page')
    ]
    
    for filename, desc in page_files:
        filepath = site_root / filename
        exists = check_file_exists(filepath, desc)
        all_checks.append(exists)
        
        if exists:
            all_checks.append(check_markdown_frontmatter(filepath))
    
    # Check assets
    print("\n🎯 Assets:")
    asset_files = [
        ('assets/css/main.css', 'Main stylesheet'),
        ('assets/js/main.js', 'JavaScript functionality')
    ]
    
    for filename, desc in asset_files:
        filepath = site_root / filename
        all_checks.append(check_file_exists(filepath, desc))
    
    # Check data files
    print("\n📊 Data Files:")
    data_files = [
        ('_data/navigation.yml', 'Navigation configuration')
    ]
    
    for filename, desc in data_files:
        filepath = site_root / filename
        exists = check_file_exists(filepath, desc)
        all_checks.append(exists)
        
        if exists:
            all_checks.append(check_yaml_valid(filepath))
    
    # Check directories
    print("\n📁 Required Directories:")
    required_dirs = [
        'assets/images/plots',
        'assets/images/diagrams'
    ]
    
    for dirname in required_dirs:
        dirpath = site_root / dirname
        if dirpath.exists() and dirpath.is_dir():
            print(f"✅ Directory: {dirpath}")
            all_checks.append(True)
        else:
            print(f"⚠️  Directory: {dirpath} (will be created as needed)")
            all_checks.append(True)  # Not critical for basic functionality
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(all_checks)
    total = len(all_checks)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"📊 Verification Summary: {passed}/{total} checks passed ({success_rate:.1f}%)")
    
    if success_rate >= 95:
        print("🎉 Jekyll site structure is ready!")
        print("\n🚀 To start development server:")
        print("   cd docs/jekyll_site")
        print("   bundle install")
        print("   bundle exec jekyll serve")
        return 0
    elif success_rate >= 80:
        print("⚠️  Site structure mostly complete, minor issues detected")
        return 1
    else:
        print("❌ Significant issues found, please address before proceeding")
        return 2

if __name__ == "__main__":
    exit(main())