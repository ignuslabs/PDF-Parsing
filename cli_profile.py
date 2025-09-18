#!/usr/bin/env python3
"""
Command-line interface for profiling PDF Parser operations.

This module provides commands to profile various operations and generate
performance reports in different formats.
"""

import click
import sys
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.parser import DoclingParser
from src.core.search import SmartSearchEngine
from src.verification.renderer import PDFRenderer
from src.utils.profiling import (
    profile_context,
    ProfileAnalyzer,
    ProfileComparator,
    load_profile,
    merge_profiles
)


@click.group()
def cli():
    """PDF Parser Profiling CLI - Performance analysis tools."""
    pass


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--ocr', is_flag=True, help='Enable OCR for scanned documents')
@click.option('--tables', is_flag=True, default=True, help='Enable table extraction')
@click.option('--output', '-o', type=click.Path(), help='Output file for profile data')
@click.option('--format', '-f', type=click.Choice(['text', 'html', 'json']), default='text',
              help='Report format')
@click.option('--top-n', type=int, default=30, help='Number of top functions to show')
@click.option('--memory', is_flag=True, help='Include memory profiling')
def parse(pdf_path: str, ocr: bool, tables: bool, output: Optional[str], 
          format: str, top_n: int, memory: bool):
    """Profile PDF parsing operation."""
    pdf_path = Path(pdf_path) # type: ignore
    
    click.echo(f"Profiling parser for: {pdf_path.name}")
    click.echo(f"Options: OCR={ocr}, Tables={tables}, Memory={memory}")
    
    # Setup output file
    if output:
        output_file = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"profiles/parse_{pdf_path.stem}_{timestamp}.prof")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Profile the parsing operation
    with profile_context(
        name=f"parse_{pdf_path.name}",
        output_file=output_file,
        print_stats=False,
        memory_profile=memory
    ) as profiler:
        parser = DoclingParser(
            enable_ocr=ocr,
            enable_tables=tables,
            enable_profiling=False  # We're already profiling externally
        )
        elements = parser.parse_document(pdf_path)
    
    click.echo(f"\nParsed {len(elements)} elements")
    
    # Generate report
    stats = load_profile(output_file)
    analyzer = ProfileAnalyzer(stats)
    
    # Save report in requested format
    report_file = output_file.with_suffix(f'.{format}')
    analyzer.save_report(report_file, format=format, top_n=top_n)
    
    # Print summary to console
    click.echo("\n" + "="*60)
    click.echo("TOP PERFORMANCE BOTTLENECKS")
    click.echo("="*60)
    
    top_functions = analyzer.get_top_functions(10, sort_by='cumulative')
    for i, stat in enumerate(top_functions, 1):
        func_name = stat.function_name.split('/')[-1] if '/' in stat.function_name else stat.function_name
        click.echo(f"{i:2}. {func_name[:60]:<60}")
        click.echo(f"    Calls: {stat.calls:,} | Time: {stat.cumulative_time:.3f}s | "
                  f"Per call: {stat.time_per_call*1000:.2f}ms")
    
    click.echo(f"\nProfile data saved to: {output_file}")
    click.echo(f"Report saved to: {report_file}")


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.argument('query')
@click.option('--fuzzy/--exact', default=True, help='Enable fuzzy matching')
@click.option('--output', '-o', type=click.Path(), help='Output file for profile data')
@click.option('--memory', is_flag=True, help='Include memory profiling')
def search(pdf_path: str, query: str, fuzzy: bool, output: Optional[str], memory: bool):
    """Profile search operation on parsed document."""
    pdf_path = Path(pdf_path)
    
    click.echo(f"Profiling search in: {pdf_path.name}")
    click.echo(f"Query: '{query}' (fuzzy={fuzzy})")
    
    # Setup output file
    if output:
        output_file = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"profiles/search_{timestamp}.prof")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # First parse the document (not profiled)
    click.echo("Parsing document...")
    parser = DoclingParser()
    elements = parser.parse_document(pdf_path)
    
    # Profile the search operation
    click.echo(f"Profiling search...")
    with profile_context(
        name=f"search_{query}",
        output_file=output_file,
        print_stats=False,
        memory_profile=memory
    ):
        search_engine = SmartSearchEngine(elements)
        results = search_engine.search(query, enable_fuzzy=fuzzy)
    
    click.echo(f"\nFound {len(results)} results")
    
    # Generate report
    stats = load_profile(output_file)
    analyzer = ProfileAnalyzer(stats)
    
    # Print summary
    click.echo("\n" + "="*60)
    click.echo("SEARCH PERFORMANCE ANALYSIS")
    click.echo("="*60)
    
    top_functions = analyzer.get_top_functions(10, sort_by='time')
    for i, stat in enumerate(top_functions, 1):
        func_name = stat.function_name.split('/')[-1] if '/' in stat.function_name else stat.function_name
        click.echo(f"{i:2}. {func_name[:60]:<60}")
        click.echo(f"    Time: {stat.total_time:.3f}s | Calls: {stat.calls:,}")
    
    click.echo(f"\nProfile saved to: {output_file}")


@cli.command()
@click.argument('pdf_paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for profile data')
@click.option('--parallel/--sequential', default=False, help='Process in parallel')
@click.option('--memory', is_flag=True, help='Include memory profiling')
def batch(pdf_paths: tuple, output: Optional[str], parallel: bool, memory: bool):
    """Profile batch processing of multiple PDFs."""
    pdf_files = [Path(p) for p in pdf_paths]
    
    click.echo(f"Profiling batch processing of {len(pdf_files)} files")
    click.echo(f"Mode: {'Parallel' if parallel else 'Sequential'}")
    
    # Setup output file
    if output:
        output_file = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"profiles/batch_{len(pdf_files)}files_{timestamp}.prof")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Profile batch processing
    with profile_context(
        name=f"batch_{len(pdf_files)}_files",
        output_file=output_file,
        print_stats=False,
        memory_profile=memory
    ):
        parser = DoclingParser()
        
        if parallel:
            # Parallel processing would go here
            # For now, fall back to sequential
            click.echo("Note: Parallel processing not yet implemented, using sequential")
        
        results = {}
        for pdf_file in pdf_files:
            click.echo(f"  Processing {pdf_file.name}...")
            elements = parser.parse_document(pdf_file)
            results[pdf_file] = len(elements)
    
    # Print results
    click.echo("\n" + "="*60)
    click.echo("BATCH PROCESSING RESULTS")
    click.echo("="*60)
    
    total_elements = sum(results.values())
    click.echo(f"Total files processed: {len(results)}")
    click.echo(f"Total elements extracted: {total_elements:,}")
    click.echo(f"Average elements per file: {total_elements/len(results):.1f}")
    
    # Generate performance report
    stats = load_profile(output_file)
    analyzer = ProfileAnalyzer(stats)
    
    report_file = output_file.with_suffix('.html')
    analyzer.save_report(report_file, format='html', top_n=50)
    
    click.echo(f"\nProfile saved to: {output_file}")
    click.echo(f"Report saved to: {report_file}")


@cli.command()
@click.argument('profile_file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['text', 'html', 'json']), default='text',
              help='Report format')
@click.option('--top-n', type=int, default=30, help='Number of top functions')
@click.option('--sort', type=click.Choice(['cumulative', 'time', 'calls']), default='cumulative',
              help='Sort criteria')
@click.option('--output', '-o', type=click.Path(), help='Output file for report')
def analyze(profile_file: str, format: str, top_n: int, sort: str, output: Optional[str]):
    """Analyze existing profile data and generate report."""
    profile_path = Path(profile_file)
    
    click.echo(f"Analyzing profile: {profile_path.name}")
    
    # Load profile data
    stats = load_profile(profile_path)
    analyzer = ProfileAnalyzer(stats)
    
    # Generate report
    if output:
        report_file = Path(output)
    else:
        report_file = profile_path.with_suffix(f'.{format}')
    
    analyzer.save_report(report_file, format=format, top_n=top_n)
    
    # Also print to console
    if format == 'text':
        report_content = analyzer.generate_text_report(top_n)
        click.echo(report_content)
    else:
        # Print summary for non-text formats
        click.echo("\n" + "="*60)
        click.echo("TOP FUNCTIONS BY " + sort.upper())
        click.echo("="*60)
        
        top_functions = analyzer.get_top_functions(min(10, top_n), sort_by=sort)
        for i, stat in enumerate(top_functions, 1):
            func_name = stat.function_name.split('/')[-1] if '/' in stat.function_name else stat.function_name
            click.echo(f"{i:2}. {func_name[:60]:<60}")
            
            if sort == 'cumulative':
                click.echo(f"    Cumulative: {stat.cumulative_time:.3f}s | "
                          f"Calls: {stat.calls:,}")
            elif sort == 'time':
                click.echo(f"    Total: {stat.total_time:.3f}s | "
                          f"Per call: {stat.time_per_call*1000:.2f}ms")
            else:  # calls
                click.echo(f"    Calls: {stat.calls:,} | "
                          f"Total: {stat.total_time:.3f}s")
    
    click.echo(f"\nReport saved to: {report_file}")


@cli.command()
@click.argument('baseline', type=click.Path(exists=True))
@click.argument('comparison', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for comparison report')
def compare(baseline: str, comparison: str, output: Optional[str]):
    """Compare two profiling runs to identify performance changes."""
    baseline_path = Path(baseline)
    comparison_path = Path(comparison)
    
    click.echo(f"Comparing profiles:")
    click.echo(f"  Baseline: {baseline_path.name}")
    click.echo(f"  Comparison: {comparison_path.name}")
    
    # Create comparator
    comparator = ProfileComparator(baseline_path, comparison_path)
    
    # Generate comparison report
    report = comparator.generate_comparison_report()
    
    # Print to console
    click.echo("\n" + report)
    
    # Save to file if requested
    if output:
        output_path = Path(output)
        output_path.write_text(report)
        click.echo(f"\nComparison report saved to: {output_path}")
    
    # Also generate detailed JSON comparison
    comparison_data = comparator.compare_top_functions(30)
    
    # Find biggest improvements and regressions
    sorted_by_change = sorted(
        comparison_data.items(),
        key=lambda x: x[1]['time_difference']
    )
    
    if sorted_by_change:
        click.echo("\n" + "="*60)
        click.echo("BIGGEST IMPROVEMENTS")
        click.echo("="*60)
        
        for func, data in sorted_by_change[:5]:
            if data['time_difference'] < -0.001:
                func_short = func.split('/')[-1] if '/' in func else func
                click.echo(f"✓ {func_short[:60]}")
                click.echo(f"  {data['percent_change']:+.1f}% faster "
                          f"({data['baseline_time']:.3f}s → {data['comparison_time']:.3f}s)")
        
        click.echo("\n" + "="*60)
        click.echo("BIGGEST REGRESSIONS")
        click.echo("="*60)
        
        for func, data in reversed(sorted_by_change[-5:]):
            if data['time_difference'] > 0.001:
                func_short = func.split('/')[-1] if '/' in func else func
                click.echo(f"✗ {func_short[:60]}")
                click.echo(f"  {data['percent_change']:+.1f}% slower "
                          f"({data['baseline_time']:.3f}s → {data['comparison_time']:.3f}s)")


@cli.command()
@click.argument('profile_files', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file for merged profile')
def merge(profile_files: tuple, output: str):
    """Merge multiple profile files into one."""
    profile_paths = [Path(p) for p in profile_files]
    output_path = Path(output)
    
    click.echo(f"Merging {len(profile_paths)} profile files")
    
    # Merge profiles
    merged_stats = merge_profiles(*profile_paths)
    
    # Save merged profile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_stats.dump_stats(str(output_path))
    
    # Generate summary
    analyzer = ProfileAnalyzer(merged_stats)
    
    click.echo("\n" + "="*60)
    click.echo("MERGED PROFILE SUMMARY")
    click.echo("="*60)
    
    top_functions = analyzer.get_top_functions(10)
    for i, stat in enumerate(top_functions, 1):
        func_name = stat.function_name.split('/')[-1] if '/' in stat.function_name else stat.function_name
        click.echo(f"{i:2}. {func_name[:60]:<60}")
        click.echo(f"    Cumulative: {stat.cumulative_time:.3f}s | Calls: {stat.calls:,}")
    
    click.echo(f"\nMerged profile saved to: {output_path}")
    
    # Also save HTML report
    report_file = output_path.with_suffix('.html')
    analyzer.save_report(report_file, format='html', top_n=50)
    click.echo(f"HTML report saved to: {report_file}")


if __name__ == '__main__':
    cli()