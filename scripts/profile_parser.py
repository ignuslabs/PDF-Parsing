#!/usr/bin/env python3
"""
Profile PDF parser performance with various configurations.

This script profiles the DoclingParser with different settings to identify
performance bottlenecks and optimization opportunities.
"""

import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.parser import DoclingParser
from src.utils.profiling import profile_context, ProfileAnalyzer, load_profile
from src.utils.logging_config import setup_logging


def profile_basic_parsing(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Profile basic PDF parsing without any extras."""
    print(f"\n{'='*60}")
    print("Profiling: Basic Parsing (no OCR, no tables)")
    print(f"{'='*60}")
    
    output_file = output_dir / "profile_basic.prof"
    
    with profile_context(
        name="basic_parsing",
        output_file=output_file,
        print_stats=False,
        memory_profile=True
    ):
        parser = DoclingParser(
            enable_ocr=False,
            enable_tables=False,
            generate_page_images=False,
            enable_kv_extraction=False,
            header_classifier_enabled=False,
            enable_form_fields=False
        )
        elements = parser.parse_document(pdf_path)
    
    # Analyze results
    stats = load_profile(output_file)
    analyzer = ProfileAnalyzer(stats)
    top_functions = analyzer.get_top_functions(5, sort_by='cumulative')
    
    result = {
        'config': 'basic',
        'elements_count': len(elements),
        'profile_file': str(output_file),
        'top_bottlenecks': [
            {
                'function': stat.function_name.split('/')[-1][:60],
                'cumulative_time': stat.cumulative_time,
                'calls': stat.calls
            }
            for stat in top_functions
        ]
    }
    
    print(f"Elements extracted: {len(elements)}")
    print(f"Profile saved to: {output_file}")
    
    return result


def profile_with_tables(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Profile parsing with table extraction enabled."""
    print(f"\n{'='*60}")
    print("Profiling: With Table Extraction")
    print(f"{'='*60}")
    
    output_file = output_dir / "profile_tables.prof"
    
    with profile_context(
        name="table_parsing",
        output_file=output_file,
        print_stats=False,
        memory_profile=True
    ):
        parser = DoclingParser(
            enable_ocr=False,
            enable_tables=True,
            generate_page_images=False,
            table_mode='accurate'
        )
        elements = parser.parse_document(pdf_path)
    
    # Count table elements
    table_elements = [e for e in elements if e.element_type == 'table']
    
    stats = load_profile(output_file)
    analyzer = ProfileAnalyzer(stats)
    top_functions = analyzer.get_top_functions(5, sort_by='cumulative')
    
    result = {
        'config': 'with_tables',
        'elements_count': len(elements),
        'table_count': len(table_elements),
        'profile_file': str(output_file),
        'top_bottlenecks': [
            {
                'function': stat.function_name.split('/')[-1][:60],
                'cumulative_time': stat.cumulative_time,
                'calls': stat.calls
            }
            for stat in top_functions
        ]
    }
    
    print(f"Elements extracted: {len(elements)} (Tables: {len(table_elements)})")
    print(f"Profile saved to: {output_file}")
    
    return result


def profile_with_ocr(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Profile parsing with OCR enabled."""
    print(f"\n{'='*60}")
    print("Profiling: With OCR (Tesseract)")
    print(f"{'='*60}")
    
    output_file = output_dir / "profile_ocr.prof"
    
    with profile_context(
        name="ocr_parsing",
        output_file=output_file,
        print_stats=False,
        memory_profile=True
    ):
        parser = DoclingParser(
            enable_ocr=True,
            enable_tables=False,
            generate_page_images=False,
            ocr_engine='tesseract',
            ocr_lang='eng'
        )
        elements = parser.parse_document(pdf_path)
    
    stats = load_profile(output_file)
    analyzer = ProfileAnalyzer(stats)
    top_functions = analyzer.get_top_functions(5, sort_by='cumulative')
    
    result = {
        'config': 'with_ocr',
        'elements_count': len(elements),
        'profile_file': str(output_file),
        'top_bottlenecks': [
            {
                'function': stat.function_name.split('/')[-1][:60],
                'cumulative_time': stat.cumulative_time,
                'calls': stat.calls
            }
            for stat in top_functions
        ]
    }
    
    print(f"Elements extracted: {len(elements)}")
    print(f"Profile saved to: {output_file}")
    
    return result


def profile_full_features(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Profile parsing with all features enabled."""
    print(f"\n{'='*60}")
    print("Profiling: Full Features (OCR + Tables + Images + KV)")
    print(f"{'='*60}")
    
    output_file = output_dir / "profile_full.prof"
    
    with profile_context(
        name="full_parsing",
        output_file=output_file,
        print_stats=False,
        memory_profile=True
    ):
        parser = DoclingParser(
            enable_ocr=True,
            enable_tables=True,
            generate_page_images=True,
            enable_kv_extraction=True,
            header_classifier_enabled=True,
            enable_form_fields=True,
            ocr_engine='tesseract',
            table_mode='accurate'
        )
        elements = parser.parse_document(pdf_path)
    
    stats = load_profile(output_file)
    analyzer = ProfileAnalyzer(stats)
    top_functions = analyzer.get_top_functions(5, sort_by='cumulative')
    
    result = {
        'config': 'full_features',
        'elements_count': len(elements),
        'profile_file': str(output_file),
        'top_bottlenecks': [
            {
                'function': stat.function_name.split('/')[-1][:60],
                'cumulative_time': stat.cumulative_time,
                'calls': stat.calls
            }
            for stat in top_functions
        ]
    }
    
    print(f"Elements extracted: {len(elements)}")
    print(f"Profile saved to: {output_file}")
    
    return result


def profile_with_profiling_flag(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Profile parsing with internal profiling enabled."""
    print(f"\n{'='*60}")
    print("Profiling: With Internal Profiling Flag")
    print(f"{'='*60}")
    
    output_file = output_dir / "profile_internal.prof"
    
    # This will use the parser's internal profiling
    parser = DoclingParser(
        enable_ocr=False,
        enable_tables=True,
        enable_profiling=True  # Enable internal profiling
    )
    
    # The parser will profile itself
    elements = parser.parse_document(pdf_path)
    
    result = {
        'config': 'internal_profiling',
        'elements_count': len(elements),
        'note': 'Used parser internal profiling'
    }
    
    print(f"Elements extracted: {len(elements)}")
    print("Note: Profile output shown in console above")
    
    return result


def generate_comparison_report(results: List[Dict[str, Any]], output_dir: Path):
    """Generate a comparison report of all profiling runs."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    # Sort by configuration complexity
    config_order = ['basic', 'with_tables', 'with_ocr', 'full_features', 'internal_profiling']
    results_sorted = sorted(results, key=lambda x: config_order.index(x['config']) 
                           if x['config'] in config_order else 999)
    
    # Print comparison table
    print(f"{'Configuration':<20} {'Elements':<10} {'Top Bottleneck':<40}")
    print("-" * 70)
    
    for result in results_sorted:
        config = result['config']
        elements = result['elements_count']
        
        if 'top_bottlenecks' in result and result['top_bottlenecks']:
            top_bottleneck = result['top_bottlenecks'][0]['function']
            top_time = result['top_bottlenecks'][0]['cumulative_time']
            bottleneck_str = f"{top_bottleneck[:35]} ({top_time:.2f}s)"
        else:
            bottleneck_str = "N/A"
        
        print(f"{config:<20} {elements:<10} {bottleneck_str:<40}")
    
    # Save JSON report
    report_file = output_dir / "profiling_comparison.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Profile PDF parser performance')
    parser.add_argument('pdf_path', type=Path, help='Path to PDF file')
    parser.add_argument('--output-dir', type=Path, default=Path('profiles/parser'),
                       help='Directory for profile outputs')
    parser.add_argument('--configs', nargs='+', 
                       choices=['basic', 'tables', 'ocr', 'full', 'internal', 'all'],
                       default=['all'],
                       help='Configurations to profile')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        setup_logging(level='DEBUG')
    else:
        setup_logging(level='INFO')
    
    # Validate PDF exists
    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Profiling PDF: {args.pdf_path.name}")
    print(f"Output directory: {args.output_dir}")
    
    # Run profiling based on selected configs
    results = []
    
    if 'all' in args.configs or 'basic' in args.configs:
        results.append(profile_basic_parsing(args.pdf_path, args.output_dir))
    
    if 'all' in args.configs or 'tables' in args.configs:
        results.append(profile_with_tables(args.pdf_path, args.output_dir))
    
    if 'all' in args.configs or 'ocr' in args.configs:
        results.append(profile_with_ocr(args.pdf_path, args.output_dir))
    
    if 'all' in args.configs or 'full' in args.configs:
        results.append(profile_full_features(args.pdf_path, args.output_dir))
    
    if 'all' in args.configs or 'internal' in args.configs:
        results.append(profile_with_profiling_flag(args.pdf_path, args.output_dir))
    
    # Generate comparison report
    if len(results) > 1:
        generate_comparison_report(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Profiling complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()