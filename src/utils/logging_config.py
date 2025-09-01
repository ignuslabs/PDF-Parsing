"""
Centralized logging configuration for Smart PDF Parser.

This module provides comprehensive logging setup with:
- Color-coded console output
- File logging with rotation
- Progress tracking utilities
- Performance monitoring
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Any, Dict, Callable
from datetime import datetime
import os
from functools import wraps


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def format(self, record):
        # Add color to levelname
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


class ProgressTracker:
    """Track progress of multi-step operations with timing and ETA estimation."""
    
    def __init__(self, name: str, total_steps: Optional[int] = None, logger: Optional[logging.Logger] = None):
        self.name = name
        self.total_steps = total_steps
        self.logger = logger or logging.getLogger(__name__)
        
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.step_names = []
        self.completed_steps = []
        
        self.logger.info(f"🚀 Starting {name}")
    
    def start_step(self, step_name: str) -> None:
        """Start a new step in the process."""
        self.current_step += 1
        self.step_names.append(step_name)
        step_start_time = time.time()
        self.step_times.append(step_start_time)
        
        progress_info = ""
        if self.total_steps:
            percentage = (self.current_step / self.total_steps) * 100
            progress_info = f" [{self.current_step}/{self.total_steps}] ({percentage:.1f}%)"
        
        self.logger.info(f"⏳ Step {self.current_step}{progress_info}: {step_name}")
    
    def complete_step(self, result: Any = None, details: Optional[str] = None) -> None:
        """Mark current step as completed with optional result details."""
        if not self.step_times:
            self.logger.warning("No active step to complete")
            return
        
        step_duration = time.time() - self.step_times[-1]
        step_name = self.step_names[-1] if self.step_names else f"Step {self.current_step}"
        
        self.completed_steps.append({
            'name': step_name,
            'duration': step_duration,
            'result': result,
            'details': details
        })
        
        # Format completion message
        completion_msg = f"✅ Completed: {step_name} ({step_duration:.2f}s)"
        if details:
            completion_msg += f" - {details}"
        
        self.logger.info(completion_msg)
        
        # Show ETA if we have total steps
        if self.total_steps and self.current_step < self.total_steps:
            avg_step_time = sum(step['duration'] for step in self.completed_steps) / len(self.completed_steps)
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_step_time * remaining_steps
            self.logger.info(f"📊 ETA for remaining steps: {eta_seconds:.1f}s")
    
    def finish(self, success: bool = True) -> Dict[str, Any]:
        """Finish the progress tracking and return summary."""
        total_duration = time.time() - self.start_time
        
        summary = {
            'name': self.name,
            'total_duration': total_duration,
            'completed_steps': len(self.completed_steps),
            'total_steps': self.total_steps,
            'success': success,
            'step_details': self.completed_steps
        }
        
        if success:
            self.logger.info(f"🎉 Completed {self.name} in {total_duration:.2f}s "
                           f"({len(self.completed_steps)} steps)")
        else:
            self.logger.error(f"❌ Failed {self.name} after {total_duration:.2f}s")
        
        return summary


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_colors: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        enable_colors: Enable colored console output
        log_format: Custom log format string
    
    Returns:
        Configured root logger
    """
    # Get log level from environment variable or parameter
    log_level = os.getenv("PDF_PARSER_LOG_LEVEL", level).upper()
    
    # Create root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Set level
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if enable_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(log_format)
    else:
        console_formatter = logging.Formatter(log_format)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def time_it(func: Callable = None, *, logger: Optional[logging.Logger] = None):
    """
    Decorator to time function execution and log the result.
    
    Args:
        func: Function to decorate
        logger: Logger to use (defaults to function's module logger)
    
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logger or logging.getLogger(f.__module__)
            
            func_logger.debug(f"⏱️  Starting {f.__name__}")
            
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.info(f"✅ {f.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                func_logger.error(f"❌ {f.__name__} failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    
    if func is None:
        # Decorator called with arguments
        return decorator
    else:
        # Decorator called without arguments
        return decorator(func)


def log_memory_usage(logger: logging.Logger, operation: str = "operation"):
    """Log current memory usage for monitoring."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.debug(f"💾 Memory usage after {operation}: {memory_mb:.1f}MB")
    except ImportError:
        logger.debug("psutil not available - cannot log memory usage")
    except Exception as e:
        logger.debug(f"Failed to log memory usage: {e}")


def create_progress_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Create a logger specifically for progress reporting."""
    logger = logging.getLogger(f"progress.{name}")
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    return logger


# Initialize default logging on module import
if not logging.getLogger().handlers:
    setup_logging()