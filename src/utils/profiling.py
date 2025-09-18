"""
Comprehensive profiling utilities for Smart PDF Parser.

This module provides decorators, context managers, and analysis tools for
CPU and memory profiling of PDF parsing operations.
"""

import cProfile
import pstats
import io
import time
import functools
import contextlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import tracemalloc
import linecache

logger = logging.getLogger(__name__)


@dataclass
class ProfileStats:
    """Container for profiling statistics."""
    function_name: str
    total_time: float
    calls: int
    time_per_call: float
    cumulative_time: float
    memory_start: Optional[float] = None
    memory_peak: Optional[float] = None
    memory_end: Optional[float] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ProfileAnalyzer:
    """Analyze and report profiling data."""
    
    def __init__(self, profile_data: pstats.Stats):
        """Initialize analyzer with profile statistics.
        
        Args:
            profile_data: pstats.Stats object from cProfile
        """
        self.stats = profile_data
        self.stats.strip_dirs()
        
    def get_top_functions(self, n: int = 20, sort_by: str = 'cumulative') -> List[ProfileStats]:
        """Get top N functions by specified metric.
        
        Args:
            n: Number of top functions to return
            sort_by: Sort metric ('cumulative', 'time', 'calls')
            
        Returns:
            List of ProfileStats objects
        """
        self.stats.sort_stats(sort_by)
        
        results = []
        for (filename, line_num, func_name), (cc, nc, tt, ct, callers) in \
                list(self.stats.stats.items())[:n]:
            stat = ProfileStats(
                function_name=f"{filename}:{line_num}({func_name})",
                total_time=tt,
                calls=nc,
                time_per_call=tt/nc if nc > 0 else 0,
                cumulative_time=ct,
                timestamp=datetime.now().isoformat()
            )
            results.append(stat)
            
        return results
    
    def generate_text_report(self, top_n: int = 30) -> str:
        """Generate human-readable text report.
        
        Args:
            top_n: Number of top functions to include
            
        Returns:
            Formatted text report
        """
        output = io.StringIO()
        output.write("=" * 80 + "\n")
        output.write("PROFILING REPORT\n")
        output.write("=" * 80 + "\n\n")
        
        # Summary statistics
        output.write("Summary Statistics:\n")
        output.write("-" * 40 + "\n")
        total_calls = sum(stat[1] for stat in self.stats.stats.values())
        total_time = sum(stat[2] for stat in self.stats.stats.values())
        output.write(f"Total function calls: {total_calls:,}\n")
        output.write(f"Total execution time: {total_time:.4f} seconds\n\n")
        
        # Top functions by cumulative time
        output.write(f"Top {top_n} Functions by Cumulative Time:\n")
        output.write("-" * 40 + "\n")
        self.stats.sort_stats('cumulative')
        self.stats.print_stats(top_n, output)
        
        # Top functions by total time
        output.write(f"\nTop {top_n} Functions by Total Time:\n")
        output.write("-" * 40 + "\n")
        self.stats.sort_stats('time')
        self.stats.print_stats(top_n, output)
        
        # Callers information
        output.write(f"\nTop {top_n//2} Function Callers:\n")
        output.write("-" * 40 + "\n")
        self.stats.print_callers(top_n//2, output)
        
        return output.getvalue()
    
    def generate_html_report(self, top_n: int = 50) -> str:
        """Generate HTML report with tables and charts.
        
        Args:
            top_n: Number of top functions to include
            
        Returns:
            HTML report as string
        """
        top_functions = self.get_top_functions(top_n)
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Profiling Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: bold; }
                tr:hover { background-color: #f5f5f5; }
                .metric { font-weight: bold; color: #0066cc; }
                .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>PDF Parser Profiling Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Functions Profiled: <span class="metric">{total_functions}</span></p>
                <p>Total Execution Time: <span class="metric">{total_time:.4f}</span> seconds</p>
            </div>
            
            <h2>Top {top_n} Functions by Cumulative Time</h2>
            <table>
                <tr>
                    <th>Function</th>
                    <th>Calls</th>
                    <th>Total Time (s)</th>
                    <th>Time per Call (ms)</th>
                    <th>Cumulative Time (s)</th>
                </tr>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_functions=len(self.stats.stats),
            total_time=sum(stat[2] for stat in self.stats.stats.values()),
            top_n=top_n
        )
        
        for stat in top_functions:
            # Truncate long function names for readability
            func_name = stat.function_name
            if len(func_name) > 80:
                func_name = "..." + func_name[-77:]
                
            html += f"""
                <tr>
                    <td>{func_name}</td>
                    <td>{stat.calls:,}</td>
                    <td>{stat.total_time:.4f}</td>
                    <td>{stat.time_per_call * 1000:.2f}</td>
                    <td>{stat.cumulative_time:.4f}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def save_report(self, filepath: Path, format: str = 'text', top_n: int = 30):
        """Save profiling report to file.
        
        Args:
            filepath: Path to save report
            format: Report format ('text', 'html', 'json')
            top_n: Number of top functions to include
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'html':
            content = self.generate_html_report(top_n)
        elif format == 'json':
            data = {
                'timestamp': datetime.now().isoformat(),
                'top_functions': [stat.to_dict() for stat in self.get_top_functions(top_n)]
            }
            content = json.dumps(data, indent=2)
        else:  # Default to text
            content = self.generate_text_report(top_n)
        
        filepath.write_text(content)
        logger.info(f"Profiling report saved to {filepath}")


def profile_function(
    output_file: Optional[Path] = None,
    sort_by: str = 'cumulative',
    print_stats: bool = True,
    memory_profile: bool = False
) -> Callable:
    """Decorator to profile a function's execution.
    
    Args:
        output_file: Optional file to save profiling data
        sort_by: Sort metric for statistics
        print_stats: Whether to print stats to console
        memory_profile: Whether to include memory profiling
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start memory tracking if requested
            if memory_profile:
                tracemalloc.start()
                snapshot_start = tracemalloc.take_snapshot()
            
            # CPU profiling
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                
                # Process CPU stats
                stats = pstats.Stats(profiler)
                analyzer = ProfileAnalyzer(stats)
                
                if print_stats:
                    print(f"\n{'='*60}")
                    print(f"Profile for: {func.__name__}")
                    print(f"{'='*60}")
                    stats.sort_stats(sort_by)
                    stats.print_stats(20)
                
                # Memory profiling
                if memory_profile:
                    snapshot_end = tracemalloc.take_snapshot()
                    top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
                    
                    print(f"\n{'='*60}")
                    print(f"Memory Profile for: {func.__name__}")
                    print(f"{'='*60}")
                    print("[ Top 10 Memory Allocations ]")
                    for stat in top_stats[:10]:
                        print(stat)
                    
                    tracemalloc.stop()
                
                # Save to file if requested
                if output_file:
                    profiler.dump_stats(str(output_file))
                    logger.info(f"Profile data saved to {output_file}")
                    
                    # Also save readable report
                    report_file = output_file.with_suffix('.txt')
                    analyzer.save_report(report_file, format='text')
            
            return result
        
        return wrapper
    return decorator


@contextlib.contextmanager
def profile_context(
    name: str = "code_block",
    output_file: Optional[Path] = None,
    print_stats: bool = True,
    memory_profile: bool = False
):
    """Context manager for profiling code blocks.
    
    Args:
        name: Name for the profiled code block
        output_file: Optional file to save profiling data
        print_stats: Whether to print stats to console
        memory_profile: Whether to include memory profiling
        
    Example:
        with profile_context("pdf_parsing"):
            parser.parse_document(pdf_path)
    """
    # Start memory tracking if requested
    if memory_profile:
        tracemalloc.start()
        mem_start = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
    
    # Start CPU profiling
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    
    try:
        yield profiler
    finally:
        # Stop profiling
        profiler.disable()
        elapsed_time = time.time() - start_time
        
        # Process stats
        stats = pstats.Stats(profiler)
        analyzer = ProfileAnalyzer(stats)
        
        if print_stats:
            print(f"\n{'='*60}")
            print(f"Profile for: {name}")
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            print(f"{'='*60}")
            stats.sort_stats('cumulative')
            stats.print_stats(20)
        
        # Memory stats
        if memory_profile:
            mem_current, mem_peak = tracemalloc.get_traced_memory()
            mem_current_mb = mem_current / 1024 / 1024
            mem_peak_mb = mem_peak / 1024 / 1024
            
            print(f"\nMemory Usage:")
            print(f"  Start: {mem_start:.2f} MB")
            print(f"  Current: {mem_current_mb:.2f} MB")
            print(f"  Peak: {mem_peak_mb:.2f} MB")
            print(f"  Allocated: {mem_current_mb - mem_start:.2f} MB")
            
            tracemalloc.stop()
        
        # Save to file
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(output_file))
            
            # Save readable report
            report_file = output_file.with_suffix('.html')
            analyzer.save_report(report_file, format='html', top_n=50)
            
            logger.info(f"Profile saved to {output_file} and {report_file}")


class ProfileComparator:
    """Compare multiple profiling runs to identify performance changes."""
    
    def __init__(self, baseline_file: Path, comparison_file: Path):
        """Initialize comparator with two profile files.
        
        Args:
            baseline_file: Path to baseline profile data
            comparison_file: Path to comparison profile data
        """
        self.baseline = pstats.Stats(str(baseline_file))
        self.comparison = pstats.Stats(str(comparison_file))
        
        self.baseline.strip_dirs()
        self.comparison.strip_dirs()
    
    def compare_top_functions(self, n: int = 20) -> Dict[str, Dict[str, float]]:
        """Compare top N functions between runs.
        
        Args:
            n: Number of top functions to compare
            
        Returns:
            Dictionary with comparison data
        """
        baseline_analyzer = ProfileAnalyzer(self.baseline)
        comparison_analyzer = ProfileAnalyzer(self.comparison)
        
        baseline_top = {stat.function_name: stat for stat in baseline_analyzer.get_top_functions(n)}
        comparison_top = {stat.function_name: stat for stat in comparison_analyzer.get_top_functions(n)}
        
        results = {}
        all_functions = set(baseline_top.keys()) | set(comparison_top.keys())
        
        for func in all_functions:
            baseline_stat = baseline_top.get(func)
            comparison_stat = comparison_top.get(func)
            
            if baseline_stat and comparison_stat:
                time_diff = comparison_stat.cumulative_time - baseline_stat.cumulative_time
                time_diff_pct = (time_diff / baseline_stat.cumulative_time) * 100 if baseline_stat.cumulative_time > 0 else 0
                
                results[func] = {
                    'baseline_time': baseline_stat.cumulative_time,
                    'comparison_time': comparison_stat.cumulative_time,
                    'time_difference': time_diff,
                    'percent_change': time_diff_pct,
                    'baseline_calls': baseline_stat.calls,
                    'comparison_calls': comparison_stat.calls
                }
        
        return results
    
    def generate_comparison_report(self) -> str:
        """Generate a comparison report between two profiling runs.
        
        Returns:
            Formatted comparison report
        """
        comparisons = self.compare_top_functions(30)
        
        # Sort by absolute time difference
        sorted_comparisons = sorted(
            comparisons.items(),
            key=lambda x: abs(x[1]['time_difference']),
            reverse=True
        )
        
        report = ["=" * 80]
        report.append("PERFORMANCE COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Improvements
        report.append("Top Improvements:")
        report.append("-" * 40)
        improvements = [(func, data) for func, data in sorted_comparisons 
                       if data['time_difference'] < -0.01][:10]
        
        for func, data in improvements:
            report.append(f"{func[:60]:<60}")
            report.append(f"  {data['baseline_time']:.4f}s -> {data['comparison_time']:.4f}s "
                         f"({data['percent_change']:+.1f}%)")
        
        # Regressions  
        report.append("\nTop Regressions:")
        report.append("-" * 40)
        regressions = [(func, data) for func, data in sorted_comparisons 
                      if data['time_difference'] > 0.01][:10]
        
        for func, data in regressions:
            report.append(f"{func[:60]:<60}")
            report.append(f"  {data['baseline_time']:.4f}s -> {data['comparison_time']:.4f}s "
                         f"({data['percent_change']:+.1f}%)")
        
        return "\n".join(report)


def load_profile(filepath: Path) -> pstats.Stats:
    """Load profile data from file.
    
    Args:
        filepath: Path to profile data file
        
    Returns:
        pstats.Stats object
    """
    return pstats.Stats(str(filepath))


def merge_profiles(*filepaths: Path) -> pstats.Stats:
    """Merge multiple profile files into one.
    
    Args:
        *filepaths: Paths to profile files to merge
        
    Returns:
        Merged pstats.Stats object
    """
    if not filepaths:
        raise ValueError("At least one profile file must be provided")
    
    stats = pstats.Stats(str(filepaths[0]))
    for filepath in filepaths[1:]:
        stats.add(str(filepath))
    
    return stats