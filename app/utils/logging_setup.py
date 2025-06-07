"""
Logging Setup Utility
Configures application logging with proper formatting and file rotation
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_dir=None):
    """
    Setup application logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to ~/.dataplatform/logs)
    """
    
    # Create log directory if not specified
    if log_dir is None:
        log_dir = Path.home() / ".dataplatform" / "logs"
    else:
        log_dir = Path(log_dir)
        
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs with rotation
    log_file = log_dir / "dataplatform.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors only
    error_log_file = log_dir / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    setup_component_loggers()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info(f"Data Platform Application Started")
    logger.info(f"Log Level: {logging.getLevelName(log_level)}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info("=" * 50)

def setup_component_loggers():
    """Setup loggers for specific application components"""
    
    # Database operations
    db_logger = logging.getLogger('database')
    db_logger.setLevel(logging.INFO)
    
    # Vector search operations
    vector_logger = logging.getLogger('vector_search')
    vector_logger.setLevel(logging.INFO)
    
    # AI tool operations
    ai_logger = logging.getLogger('ai_tools')
    ai_logger.setLevel(logging.INFO)
    
    # File operations
    file_logger = logging.getLogger('file_operations')
    file_logger.setLevel(logging.INFO)
    
    # UI operations
    ui_logger = logging.getLogger('ui')
    ui_logger.setLevel(logging.WARNING)  # Less verbose for UI
    
    # Configuration
    config_logger = logging.getLogger('config')
    config_logger.setLevel(logging.INFO)
    
    # Security operations
    security_logger = logging.getLogger('security')
    security_logger.setLevel(logging.INFO)
    
    # Performance monitoring
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)

def get_logger(name):
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)

def log_performance(func):
    """
    Decorator to log function performance
    
    Usage:
        @log_performance
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('performance')
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"{func.__name__} completed in {duration:.3f} seconds")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"{func.__name__} failed after {duration:.3f} seconds: {e}")
            raise
            
    return wrapper

def log_database_operation(operation, table=None, query=None):
    """
    Log database operations with context
    
    Args:
        operation: Type of operation (SELECT, INSERT, UPDATE, DELETE, etc.)
        table: Table name involved (optional)
        query: SQL query (optional, will be truncated if too long)
    """
    logger = logging.getLogger('database')
    
    context_parts = [f"Operation: {operation}"]
    
    if table:
        context_parts.append(f"Table: {table}")
        
    if query:
        # Truncate long queries
        if len(query) > 200:
            query = query[:200] + "..."
        context_parts.append(f"Query: {query}")
        
    context = " | ".join(context_parts)
    logger.info(context)

def log_security_event(event_type, details=None, user=None):
    """
    Log security-related events
    
    Args:
        event_type: Type of security event (login, access_denied, etc.)
        details: Additional details about the event
        user: Username involved (if applicable)
    """
    logger = logging.getLogger('security')
    
    context_parts = [f"Security Event: {event_type}"]
    
    if user:
        context_parts.append(f"User: {user}")
        
    if details:
        context_parts.append(f"Details: {details}")
        
    context = " | ".join(context_parts)
    logger.warning(context)

def log_ai_operation(tool, operation, tokens_used=None, cost=None):
    """
    Log AI tool operations with usage metrics
    
    Args:
        tool: AI tool name (claude, gpt, local_model, etc.)
        operation: Type of operation (chat, embedding, etc.)
        tokens_used: Number of tokens used (optional)
        cost: Cost of operation (optional)
    """
    logger = logging.getLogger('ai_tools')
    
    context_parts = [f"AI Tool: {tool}", f"Operation: {operation}"]
    
    if tokens_used:
        context_parts.append(f"Tokens: {tokens_used}")
        
    if cost:
        context_parts.append(f"Cost: ${cost:.4f}")
        
    context = " | ".join(context_parts)
    logger.info(context)

def log_file_operation(operation, file_path, size=None, duration=None):
    """
    Log file operations
    
    Args:
        operation: Type of operation (read, write, index, etc.)
        file_path: Path to the file
        size: File size in bytes (optional)
        duration: Operation duration in seconds (optional)
    """
    logger = logging.getLogger('file_operations')
    
    context_parts = [f"File Operation: {operation}", f"Path: {file_path}"]
    
    if size:
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size // 1024} KB"
        else:
            size_str = f"{size // (1024 * 1024)} MB"
        context_parts.append(f"Size: {size_str}")
        
    if duration:
        context_parts.append(f"Duration: {duration:.3f}s")
        
    context = " | ".join(context_parts)
    logger.info(context)

def set_log_level(level):
    """
    Change the logging level at runtime
    
    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
        
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update file handlers
    for handler in root_logger.handlers:
        if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            if 'error' not in str(handler.baseFilename):  # Don't change error log level
                handler.setLevel(level)
                
    logger = logging.getLogger(__name__)
    logger.info(f"Log level changed to {logging.getLevelName(level)}")

def get_log_files():
    """
    Get list of current log files
    
    Returns:
        List[Path]: List of log file paths
    """
    log_dir = Path.home() / ".dataplatform" / "logs"
    
    if not log_dir.exists():
        return []
        
    log_files = []
    for file_path in log_dir.iterdir():
        if file_path.is_file() and file_path.suffix == '.log':
            log_files.append(file_path)
            
    return sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)

def cleanup_old_logs(days_to_keep=30):
    """
    Clean up log files older than specified days
    
    Args:
        days_to_keep: Number of days to keep logs (default: 30)
    """
    logger = logging.getLogger(__name__)
    log_dir = Path.home() / ".dataplatform" / "logs"
    
    if not log_dir.exists():
        return
        
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    removed_count = 0
    
    for file_path in log_dir.iterdir():
        if file_path.is_file() and file_path.suffix == '.log':
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old log file {file_path}: {e}")
                    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old log files")

# Context manager for temporary log level changes
class TemporaryLogLevel:
    """Context manager for temporarily changing log level"""
    
    def __init__(self, level):
        self.new_level = level
        self.old_level = None
        
    def __enter__(self):
        self.old_level = logging.getLogger().level
        set_log_level(self.new_level)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_log_level(self.old_level)
