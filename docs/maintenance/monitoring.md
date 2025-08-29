# System Monitoring

*How-to guide for monitoring Smart PDF Parser performance, resources, and system health*

## Overview

Effective monitoring of Smart PDF Parser ensures optimal performance, early detection of issues, and proactive maintenance. This guide covers performance metrics, resource monitoring, error tracking, and health checks for production deployments.

## Key Metrics to Monitor

### Application Performance Metrics

#### Document Processing Metrics

**Processing Throughput**:
```python
# Monitor documents processed per minute
docs_per_minute = total_documents_processed / runtime_minutes

# Track by document type and size
processing_rates = {
    "text_only": 15.2,      # docs/min
    "with_tables": 8.7,     # docs/min  
    "scanned_ocr": 3.1,     # docs/min
    "large_files": 1.8      # docs/min
}
```

**Processing Latency**:
```python
# Parse time distribution (percentiles)
parse_times = {
    "p50": 2.3,    # seconds
    "p90": 8.7,    # seconds  
    "p95": 15.2,   # seconds
    "p99": 45.1    # seconds
}

# Search response times
search_times = {
    "exact_search_p95": 0.05,    # seconds
    "fuzzy_search_p95": 0.15,    # seconds
    "semantic_search_p95": 0.8   # seconds
}
```

**Error Rates**:
```python
# Document parsing error rates
parsing_errors = {
    "total_attempts": 1000,
    "parsing_failures": 23,      # 2.3% error rate
    "ocr_failures": 8,           # 0.8% of total
    "memory_failures": 3,        # 0.3% of total  
    "timeout_failures": 12       # 1.2% of total
}
```

#### Search Performance Metrics

**Search Effectiveness**:
```python
# Search result relevance
search_metrics = {
    "avg_results_per_query": 4.2,
    "zero_result_queries": 0.08,     # 8% of queries
    "avg_confidence_score": 0.847,
    "user_click_through": 0.73       # 73% of results clicked
}
```

**Search Engine Load**:
```python
# Query volume and patterns
search_load = {
    "queries_per_minute": 45.2,
    "unique_queries_ratio": 0.65,    # 65% unique queries
    "cache_hit_rate": 0.42,          # 42% served from cache
    "concurrent_searches": 12         # average concurrent
}
```

### System Resource Metrics

#### Memory Usage Patterns

**Memory Consumption Monitoring**:
```bash
#!/bin/bash
# Memory monitoring script

# Monitor Python process memory
ps -p $(pgrep -f "smart-pdf-parser") -o pid,rss,vsz,pmem --no-headers

# Monitor system memory  
free -h | grep Mem

# Monitor memory by component
python -c "
import psutil
import gc

# Get process memory info
process = psutil.Process()
memory_info = process.memory_info()

print(f'RSS: {memory_info.rss / 1024 / 1024:.1f} MB')
print(f'VMS: {memory_info.vms / 1024 / 1024:.1f} MB')
print(f'Memory percent: {process.memory_percent():.1f}%')

# Python memory breakdown
print(f'GC objects: {len(gc.get_objects())}')
"
```

**Memory Usage Patterns**:
```python
# Typical memory usage by operation
memory_patterns = {
    "baseline": 150,         # MB - idle application
    "document_parsing": 400,  # MB - during PDF processing
    "ocr_processing": 800,   # MB - OCR active
    "large_document": 1200,  # MB - processing 100+ page docs
    "peak_usage": 1800       # MB - maximum observed
}

# Memory growth indicators (warning signs)
memory_warnings = {
    "gradual_increase": "> 50MB/hour",    # Memory leak indicator
    "rapid_spikes": "> 2GB peak",        # Processing large files
    "no_recovery": "No drop after processing"  # GC issues
}
```

#### CPU and I/O Metrics  

**CPU Usage Monitoring**:
```bash
# Monitor CPU usage by component
top -p $(pgrep -f "smart-pdf-parser") -b -n 1

# CPU usage breakdown
python -c "
import psutil
import time

# Monitor CPU usage over time
for i in range(5):
    cpu_percent = psutil.cpu_percent(interval=1)
    process = psutil.Process()
    process_cpu = process.cpu_percent()
    
    print(f'System CPU: {cpu_percent:.1f}%, Process CPU: {process_cpu:.1f}%')
    time.sleep(1)
"
```

**Disk I/O Patterns**:
```python
# Monitor file operations
import psutil

# Get I/O counters for process
process = psutil.Process()
io_counters = process.io_counters()

io_metrics = {
    "read_count": io_counters.read_count,
    "write_count": io_counters.write_count, 
    "read_bytes": io_counters.read_bytes / 1024 / 1024,  # MB
    "write_bytes": io_counters.write_bytes / 1024 / 1024  # MB
}

# Monitor disk space usage
disk_usage = psutil.disk_usage('/')
disk_metrics = {
    "total_gb": disk_usage.total / 1024**3,
    "used_gb": disk_usage.used / 1024**3,
    "free_gb": disk_usage.free / 1024**3,
    "usage_percent": (disk_usage.used / disk_usage.total) * 100
}
```

## Monitoring Implementation

### Application-Level Monitoring

#### Custom Metrics Collection

```python
"""
Application monitoring module.
"""

import time
import logging  
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Metrics for document processing operations."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    document_path: str = ""
    document_size_mb: float = 0.0
    elements_parsed: int = 0
    ocr_enabled: bool = False
    tables_found: int = 0
    processing_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def finish(self, success: bool = True, error: str = None) -> None:
        """Mark processing as complete."""
        self.end_time = datetime.now()
        self.processing_time_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = None
        self.process = psutil.Process()
        
    def start_processing(self, document_path: str) -> ProcessingMetrics:
        """Start monitoring document processing."""
        import os
        
        file_size = os.path.getsize(document_path) / 1024 / 1024  # MB
        
        metrics = ProcessingMetrics(
            document_path=document_path,
            document_size_mb=file_size
        )
        
        self.current_metrics = metrics
        logger.info(f"Started processing monitoring: {document_path}")
        
        return metrics
    
    def record_memory_peak(self) -> float:
        """Record current memory usage as peak."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        if self.current_metrics:
            self.current_metrics.memory_peak_mb = max(
                self.current_metrics.memory_peak_mb,
                memory_mb
            )
            
        return memory_mb
    
    def finish_processing(self, elements_count: int, success: bool = True, error: str = None) -> None:
        """Complete processing monitoring."""
        if not self.current_metrics:
            return
            
        self.current_metrics.elements_parsed = elements_count
        self.current_metrics.finish(success, error)
        
        # Add to history
        self.metrics_history.append(self.current_metrics)
        
        # Log performance summary
        metrics = self.current_metrics
        logger.info(
            f"Processing complete: {metrics.document_path} "
            f"({metrics.processing_time_seconds:.1f}s, "
            f"{metrics.elements_parsed} elements, "
            f"{metrics.memory_peak_mb:.1f}MB peak)"
        )
        
        self.current_metrics = None
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for recent period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.start_time >= cutoff
        ]
        
        if not recent_metrics:
            return {"message": "No recent processing data"}
        
        successful = [m for m in recent_metrics if m.success]
        failed = [m for m in recent_metrics if not m.success]
        
        return {
            "period_hours": hours,
            "total_processed": len(recent_metrics),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(recent_metrics),
            "avg_processing_time": sum(m.processing_time_seconds for m in successful) / len(successful) if successful else 0,
            "avg_elements_per_doc": sum(m.elements_parsed for m in successful) / len(successful) if successful else 0,
            "avg_memory_usage_mb": sum(m.memory_peak_mb for m in successful) / len(successful) if successful else 0,
            "total_documents_size_mb": sum(m.document_size_mb for m in recent_metrics)
        }

# Global monitor instance
performance_monitor = PerformanceMonitor()
```

#### Integration with Parser

```python  
"""
Enhanced parser with monitoring integration.
"""

from src.monitoring import performance_monitor
from src.core.parser import DoclingParser as BaseDoclingParser

class MonitoredDoclingParser(BaseDoclingParser):
    """DoclingParser with performance monitoring."""
    
    def parse_document(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """Parse document with performance monitoring."""
        
        # Start monitoring
        metrics = performance_monitor.start_processing(str(file_path))
        
        try:
            # Record initial memory
            performance_monitor.record_memory_peak()
            
            # Perform parsing
            elements = super().parse_document(file_path)
            
            # Record final memory peak
            performance_monitor.record_memory_peak()
            
            # Complete monitoring
            performance_monitor.finish_processing(len(elements), success=True)
            
            return elements
            
        except Exception as e:
            # Record failure
            performance_monitor.finish_processing(0, success=False, error=str(e))
            raise
```

### System-Level Monitoring

#### Health Check Endpoints

```python
"""
Health check endpoints for monitoring systems.
"""

from flask import Flask, jsonify
import psutil
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    })

@app.route('/health/detailed')
def detailed_health_check():
    """Detailed health check with system metrics."""
    
    # System metrics
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Application metrics
    process = psutil.Process()
    app_memory = process.memory_info()
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "memory_total_gb": memory.total / 1024**3,
            "memory_available_gb": memory.available / 1024**3,
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total / 1024**3,
            "disk_free_gb": disk.free / 1024**3,
            "disk_percent": (disk.used / disk.total) * 100,
            "cpu_percent": cpu_percent
        },
        "application": {
            "memory_rss_mb": app_memory.rss / 1024 / 1024,
            "memory_vms_mb": app_memory.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "threads": process.num_threads()
        },
        "performance": performance_monitor.get_performance_summary(hours=1)
    })

@app.route('/metrics')  
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    
    summary = performance_monitor.get_performance_summary(hours=24)
    process = psutil.Process()
    memory = process.memory_info()
    
    metrics = [
        f"# HELP documents_processed_total Total documents processed",
        f"# TYPE documents_processed_total counter", 
        f"documents_processed_total {summary.get('total_processed', 0)}",
        "",
        f"# HELP documents_success_rate Success rate of document processing",
        f"# TYPE documents_success_rate gauge",
        f"documents_success_rate {summary.get('success_rate', 0)}",
        "",
        f"# HELP memory_usage_bytes Current memory usage",
        f"# TYPE memory_usage_bytes gauge", 
        f"memory_usage_bytes {memory.rss}",
        "",
        f"# HELP avg_processing_time_seconds Average processing time",
        f"# TYPE avg_processing_time_seconds gauge",
        f"avg_processing_time_seconds {summary.get('avg_processing_time', 0)}",
    ]
    
    return "\n".join(metrics), 200, {"Content-Type": "text/plain"}
```

#### Log Analysis and Alerts  

**Structured Logging Configuration**:
```python
"""
Structured logging configuration for monitoring.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields
        if hasattr(record, 'document_path'):
            log_entry["document_path"] = record.document_path
        if hasattr(record, 'processing_time'):
            log_entry["processing_time"] = record.processing_time
        if hasattr(record, 'memory_usage'):
            log_entry["memory_usage"] = record.memory_usage
            
        return json.dumps(log_entry)

# Configure structured logging
def setup_monitoring_logging():
    """Setup logging for monitoring and alerting."""
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with structured format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for persistent logs
    file_handler = logging.FileHandler('logs/smart-pdf-parser.log')
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.FileHandler('logs/performance.log')
    perf_handler.setFormatter(StructuredFormatter())
    perf_logger.addHandler(perf_handler)
```

**Error Pattern Detection**:
```bash
#!/bin/bash
# Log analysis script for error pattern detection

LOG_FILE="logs/smart-pdf-parser.log"
ALERT_THRESHOLD=10

# Monitor error rates  
error_count=$(tail -n 1000 "$LOG_FILE" | grep '"level":"ERROR"' | wc -l)

if [ "$error_count" -gt "$ALERT_THRESHOLD" ]; then
    echo "ALERT: High error rate detected ($error_count errors in last 1000 log entries)"
fi

# Monitor memory warnings
memory_warnings=$(tail -n 1000 "$LOG_FILE" | grep -i "memory" | grep -i "warning\|error" | wc -l)

if [ "$memory_warnings" -gt 5 ]; then
    echo "ALERT: Memory issues detected ($memory_warnings warnings in recent logs)"
fi

# Monitor OCR failures
ocr_failures=$(tail -n 1000 "$LOG_FILE" | grep -i "ocr.*error\|ocr.*failed" | wc -l)

if [ "$ocr_failures" -gt 3 ]; then
    echo "ALERT: OCR processing issues ($ocr_failures failures detected)"
fi

# Extract recent error patterns  
echo "Recent error patterns:"
tail -n 2000 "$LOG_FILE" | grep '"level":"ERROR"' | \
    jq -r '.message' | sort | uniq -c | sort -nr | head -10
```

## Monitoring Tools Integration

### Prometheus and Grafana Setup

**Prometheus Configuration** (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'smart-pdf-parser'
    static_configs:
      - targets: ['localhost:5000']
    scrape_interval: 30s
    metrics_path: /metrics

  - job_name: 'node-exporter'  
    static_configs:
      - targets: ['localhost:9100']
```

**Grafana Dashboard Configuration**:
```json
{
  "dashboard": {
    "title": "Smart PDF Parser Monitoring",
    "panels": [
      {
        "title": "Document Processing Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(documents_processed_total[5m]) * 60",
            "legendFormat": "docs/min"
          }
        ]
      },
      {
        "title": "Success Rate",
        "type": "gauge", 
        "targets": [
          {
            "expr": "documents_success_rate * 100",
            "legendFormat": "Success %"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory MB"
          }
        ]
      },
      {
        "title": "Processing Time Distribution",
        "type": "histogram",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(processing_time_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack Integration  

**Logstash Configuration** (`logstash.conf`):
```ruby
input {
  file {
    path => "/var/log/smart-pdf-parser/*.log"
    start_position => "beginning"
    codec => json
  }
}

filter {
  # Parse JSON logs
  if [message] {
    json {
      source => "message"
    }
  }
  
  # Add processing time categorization
  if [processing_time] {
    if [processing_time] < 5 {
      mutate { add_field => { "performance_category" => "fast" } }
    } else if [processing_time] < 20 {
      mutate { add_field => { "performance_category" => "normal" } }
    } else {
      mutate { add_field => { "performance_category" => "slow" } }
    }
  }
  
  # Extract error categories
  if [level] == "ERROR" {
    grok {
      match => { "message" => "(?<error_type>OCRError|DocumentParsingError|ValidationError|MemoryError)" }
      tag_on_failure => ["_grokparsefailure_error_type"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "smart-pdf-parser-%{+YYYY.MM.dd}"
  }
}
```

**Kibana Visualizations**:
```json
{
  "visualizations": [
    {
      "title": "Error Rate Over Time",
      "type": "line",
      "query": {
        "bool": {
          "must": [
            {"term": {"level": "ERROR"}},
            {"range": {"@timestamp": {"gte": "now-24h"}}}
          ]
        }
      }
    },
    {
      "title": "Processing Time Heatmap",
      "type": "heatmap",
      "aggregations": {
        "x_axis": {"date_histogram": {"field": "@timestamp", "interval": "1h"}},
        "y_axis": {"terms": {"field": "performance_category"}},
        "value": {"count": {}}
      }
    },
    {
      "title": "Memory Usage Distribution", 
      "type": "histogram",
      "field": "memory_usage",
      "bins": 20
    }
  ]
}
```

## Alert Configuration

### Alert Rules and Thresholds

**Critical Alerts**:
```yaml
# Prometheus AlertManager rules
groups:
  - name: smart-pdf-parser-critical
    rules:
      - alert: HighErrorRate
        expr: documents_success_rate < 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Success rate is {{ $value }}%, below 90% threshold"
          
      - alert: MemoryUsageHigh
        expr: memory_usage_bytes > 2000000000  # 2GB
        for: 10m
        labels:
          severity: critical  
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanize }}B"
          
      - alert: ProcessingTimeTooHigh
        expr: avg_processing_time_seconds > 60
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Slow document processing"
          description: "Average processing time is {{ $value }}s"
```

**Warning Alerts**:
```yaml
  - name: smart-pdf-parser-warnings
    rules:
      - alert: OCRFailureRate
        expr: ocr_failure_rate > 0.15  # 15% OCR failures
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High OCR failure rate"
          
      - alert: DiskSpaceLow
        expr: (disk_free_gb / disk_total_gb) < 0.1  # Less than 10% free
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space"
```

### Notification Channels

**Slack Integration**:
```python
"""
Slack notification integration for alerts.
"""

import requests
import json
from typing import Dict, Any

class SlackNotifier:
    """Send monitoring alerts to Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert_type: str, message: str, metrics: Dict[str, Any] = None):
        """Send alert notification to Slack."""
        
        color_map = {
            "critical": "#ff0000",  # Red
            "warning": "#ffaa00",   # Orange  
            "info": "#0066cc"       # Blue
        }
        
        attachment = {
            "color": color_map.get(alert_type, "#cccccc"),
            "title": f"Smart PDF Parser Alert ({alert_type.upper()})",
            "text": message,
            "ts": int(time.time())
        }
        
        if metrics:
            fields = []
            for key, value in metrics.items():
                fields.append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": True
                })
            attachment["fields"] = fields
        
        payload = {
            "attachments": [attachment]
        }
        
        response = requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        
        return response.status_code == 200

# Usage in monitoring code
slack_notifier = SlackNotifier("https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK")

def check_and_alert():
    """Check system health and send alerts if needed."""
    
    summary = performance_monitor.get_performance_summary(hours=1)
    
    # Check success rate
    if summary.get("success_rate", 1.0) < 0.9:
        slack_notifier.send_alert(
            "critical",
            f"Document processing success rate dropped to {summary['success_rate']*100:.1f}%",
            {
                "failed_documents": summary["failed"],
                "total_processed": summary["total_processed"],
                "period": "Last 1 hour"
            }
        )
    
    # Check memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 2000:  # 2GB threshold
        slack_notifier.send_alert(
            "warning", 
            f"High memory usage detected: {memory_mb:.1f} MB",
            {"memory_usage_mb": memory_mb}
        )
```

## Performance Baselines and SLAs

### Performance Targets

**Service Level Objectives (SLOs)**:
```python
# Define SLOs for monitoring
SLOs = {
    "document_processing": {
        "success_rate": 0.95,           # 95% of documents processed successfully
        "availability": 0.99,           # 99% uptime
        "p95_processing_time": 30.0,    # 95% of docs processed in <30s
        "p99_processing_time": 120.0    # 99% of docs processed in <2min
    },
    "search_performance": {
        "p95_response_time": 0.1,       # 95% of searches in <100ms
        "p99_response_time": 0.5,       # 99% of searches in <500ms
        "cache_hit_rate": 0.4           # 40% of queries served from cache
    },
    "resource_usage": {
        "max_memory_per_doc": 1000,     # Max 1GB per document
        "memory_growth_rate": 50,       # Max 50MB/hour memory growth
        "cpu_utilization": 0.8          # Max 80% CPU utilization
    }
}
```

**Baseline Performance Data**:
```python
# Historical performance baselines
PERFORMANCE_BASELINES = {
    "document_types": {
        "text_only": {
            "avg_processing_time": 2.3,     # seconds
            "avg_memory_usage": 150,        # MB
            "avg_elements_extracted": 45
        },
        "with_tables": {
            "avg_processing_time": 8.7,     # seconds  
            "avg_memory_usage": 400,        # MB
            "avg_elements_extracted": 78
        },
        "scanned_ocr": {
            "avg_processing_time": 25.2,    # seconds
            "avg_memory_usage": 800,        # MB 
            "avg_elements_extracted": 52
        }
    },
    "document_sizes": {
        "small": {"max_pages": 5, "avg_time": 1.2, "avg_memory": 100},
        "medium": {"max_pages": 50, "avg_time": 12.5, "avg_memory": 500},
        "large": {"max_pages": 200, "avg_time": 45.8, "avg_memory": 1200}
    }
}
```

---

*This monitoring strategy provides comprehensive visibility into Smart PDF Parser performance, enabling proactive maintenance and optimization.*