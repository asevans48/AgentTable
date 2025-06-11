"""
File Utilities
Common file processing and analysis utilities
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from io import StringIO
import logging

logger = logging.getLogger(__name__)

def detect_schema(file_path: str, content: str) -> str:
    """Detect schema information for structured files"""
    try:
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            return detect_csv_schema(file_path, content)
        elif file_ext == '.json':
            return detect_json_schema(file_path, content)
        elif file_ext == '.py':
            return detect_python_schema(content)
        elif file_ext in ['.txt', '.md']:
            return detect_text_schema(content)
        else:
            return f"File type: {file_ext}"
            
    except Exception as e:
        logger.warning(f"Error detecting schema for {file_path}: {e}")
        return ""

def detect_csv_schema(file_path: Path, content: str) -> str:
    """Detect CSV schema"""
    try:
        # Get first few lines
        lines = content.split('\n')[:5]
        if not lines:
            return "Empty CSV file"
            
        # Try to detect delimiter and parse header
        sample = '\n'.join(lines)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.reader(StringIO(sample), delimiter=delimiter)
        headers = next(reader, [])
        
        if headers and len(headers) > 1:
            return f"CSV with {len(headers)} columns: {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}"
        else:
            return "CSV file (structure unclear)"
            
    except Exception:
        return "CSV file"

def detect_json_schema(file_path: Path, content: str) -> str:
    """Detect JSON schema"""
    try:
        data = json.loads(content[:1000])  # Parse first 1KB
        
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            return f"JSON object with keys: {', '.join(keys)}{'...' if len(data) > 5 else ''}"
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                keys = list(data[0].keys())[:5]
                return f"JSON array of objects with keys: {', '.join(keys)}{'...' if len(data[0]) > 5 else ''}"
            else:
                return f"JSON array with {len(data)} items"
        else:
            return f"JSON {type(data).__name__}"
            
    except Exception:
        return "JSON file"

def detect_python_schema(content: str) -> str:
    """Detect Python file schema"""
    patterns = {
        'class ': 'classes',
        'def ': 'functions',
        'import ': 'imports',
        'if __name__': 'executable script'
    }
    
    found = []
    for pattern, desc in patterns.items():
        if pattern in content:
            found.append(desc)
            
    if found:
        return f"Python code with: {', '.join(found)}"
    else:
        return "Python source code"

def detect_text_schema(content: str) -> str:
    """Detect text file schema"""
    lines = content.split('\n')
    line_count = len(lines)
    word_count = len(content.split())
    
    # Check for common patterns
    if any(line.startswith('#') for line in lines[:10]):
        return f"Markdown/documentation ({line_count} lines, {word_count} words)"
    elif content.count('\t') > content.count(' ') / 4:
        return f"Tab-separated text ({line_count} lines)"
    else:
        return f"Plain text ({line_count} lines, {word_count} words)"

def generate_fileset_name(file_path: str) -> str:
    """Generate a fileset name from file path"""
    path = Path(file_path)
    
    # Use parent directory name as fileset name
    parent_name = path.parent.name
    
    # Clean up the name
    if parent_name in ['', '.', '..']:
        parent_name = 'files'
        
    # Replace common separators with underscores
    parent_name = parent_name.replace('-', '_').replace(' ', '_')
    
    return parent_name
