"""
Vector Search Implementation
Provides vector search capabilities using local embeddings and vector databases
"""

import json
import sqlite3
import logging
import hashlib
import traceback
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from utils.file.file_utils import detect_schema, generate_fileset_name

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False
    np = None
    SentenceTransformer = None
    faiss = None

logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """Vector search engine using sentence transformers and FAISS"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings_cache = {}
        
        # Vector search configuration with absolute paths
        default_data_dir = Path.cwd() / "data"
        db_path_config = config_manager.get("vector_search.database_path", "")
        embed_path_config = config_manager.get("vector_search.embeddings_path", "")
        
        # Use defaults if paths are empty
        if not db_path_config:
            db_path_config = str(default_data_dir / "vector_search.db")
        if not embed_path_config:
            embed_path_config = str(default_data_dir / "embeddings")
        
        # Ensure we have proper file paths, not directories
        self.vector_db_path = Path(db_path_config)
        if self.vector_db_path.is_dir():
            # If it's a directory, append the default filename
            self.vector_db_path = self.vector_db_path / "vector_search.db"
        elif not self.vector_db_path.suffix:
            # If no file extension, assume it's a directory and add filename
            self.vector_db_path = self.vector_db_path / "vector_search.db"
            
        self.embeddings_path = Path(embed_path_config)
        if self.embeddings_path.is_file():
            # If it's a file, use its parent directory
            self.embeddings_path = self.embeddings_path.parent
        
        # Make paths absolute if they're relative
        if not self.vector_db_path.is_absolute():
            self.vector_db_path = Path.cwd() / self.vector_db_path
        if not self.embeddings_path.is_absolute():
            self.embeddings_path = Path.cwd() / self.embeddings_path
            
        self.model_name = config_manager.get("vector_search.model", "all-MiniLM-L6-v2")
        self.max_chunk_size = config_manager.get("vector_search.max_chunk_size", 1024)  # Larger for structured samples
        self.chunk_overlap = config_manager.get("vector_search.chunk_overlap", 100)
        self.sample_mode = config_manager.get("vector_search.sample_mode", True)  # Enable intelligent sampling
        
        # Ensure directories exist with proper permissions
        try:
            self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.embeddings_path.mkdir(parents=True, exist_ok=True)
            
            # Test write access to the database directory
            test_file = self.vector_db_path.parent / "test_write.tmp"
            test_file.touch()
            test_file.unlink()
            
            logger.info(f"Vector database path: {self.vector_db_path}")
            logger.info(f"Embeddings path: {self.embeddings_path}")
            
        except Exception as e:
            logger.error(f"Failed to create vector search directories: {e}")
            # Fallback to user's home directory
            fallback_dir = Path.home() / ".dataplatform" / "vector_search"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self.vector_db_path = fallback_dir / "vector_search.db"
            self.embeddings_path = fallback_dir / "embeddings"
            self.embeddings_path.mkdir(exist_ok=True)
            logger.warning(f"Using fallback directory: {fallback_dir}")
        
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for document metadata"""
        try:
            # Ensure the database file can be created
            if not self.vector_db_path.parent.exists():
                self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
                
            # Validate that we have a proper file path, not a directory
            if self.vector_db_path.is_dir():
                raise ValueError(f"Database path is a directory, not a file: {self.vector_db_path}")
                
            # Use absolute path for SQLite connection
            db_path_str = str(self.vector_db_path.resolve())
            
            # Check if database already exists
            db_exists = self.vector_db_path.exists()
            if db_exists:
                logger.info(f"Loading existing vector database at: {db_path_str}")
            else:
                logger.info(f"Creating new vector database at: {db_path_str}")
            
            # Test that we can create/access the database file
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            # Check if database exists and needs migration
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check if new columns exist and add them if needed
                cursor.execute("PRAGMA table_info(documents)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Add missing columns for metadata enhancement
                if 'fileset_name' not in columns:
                    cursor.execute("ALTER TABLE documents ADD COLUMN fileset_name TEXT")
                if 'fileset_description' not in columns:
                    cursor.execute("ALTER TABLE documents ADD COLUMN fileset_description TEXT")
                if 'schema_info' not in columns:
                    cursor.execute("ALTER TABLE documents ADD COLUMN schema_info TEXT")
                if 'tags' not in columns:
                    cursor.execute("ALTER TABLE documents ADD COLUMN tags TEXT")
                if 'user_description' not in columns:
                    cursor.execute("ALTER TABLE documents ADD COLUMN user_description TEXT")
                    
                logger.info("Database schema updated with new metadata columns")
                
                # Log existing database stats
                cursor.execute("SELECT COUNT(*) FROM documents WHERE indexed_at IS NOT NULL")
                indexed_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cursor.fetchone()[0]
                logger.info(f"Loaded existing database with {indexed_count} indexed documents and {chunk_count} chunks")
            else:
                # Create documents table with all columns
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        title TEXT,
                        content_preview TEXT,
                        file_type TEXT,
                        file_size INTEGER,
                        fileset_name TEXT,
                        fileset_description TEXT,
                        schema_info TEXT,
                        tags TEXT,
                        user_description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        indexed_at TIMESTAMP,
                        chunk_count INTEGER DEFAULT 0
                    )
                """)
                logger.info("Created new vector database tables")
            
            # Check and update chunks table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
            chunks_table_exists = cursor.fetchone() is not None
            
            if chunks_table_exists:
                # Check if enhanced_content column exists
                cursor.execute("PRAGMA table_info(chunks)")
                chunk_columns = [column[1] for column in cursor.fetchall()]
                
                if 'enhanced_content' not in chunk_columns:
                    cursor.execute("ALTER TABLE chunks ADD COLUMN enhanced_content TEXT")
                    logger.info("Added enhanced_content column to chunks table")
            else:
                # Create chunks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER,
                        chunk_index INTEGER,
                        content TEXT NOT NULL,
                        enhanced_content TEXT,
                        embedding_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id)
                    )
                """)
            
            # Create filesets table for dataset management
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS filesets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    schema_info TEXT,
                    tags TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create search history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    search_type TEXT,
                    results_count INTEGER,
                    search_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing vector database at {self.vector_db_path}: {e}")
            
            # Try fallback location if the configured path fails
            try:
                fallback_dir = Path.home() / ".dataplatform" / "vector_search"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                self.vector_db_path = fallback_dir / "vector_search.db"
                self.embeddings_path = fallback_dir / "embeddings"
                self.embeddings_path.mkdir(exist_ok=True)
                
                logger.warning(f"Using fallback database location: {self.vector_db_path}")
                
                # Try to initialize with fallback location
                db_path_str = str(self.vector_db_path.resolve())
                conn = sqlite3.connect(db_path_str)
                cursor = conn.cursor()
                
                # Create tables (same as above)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        title TEXT,
                        content_preview TEXT,
                        file_type TEXT,
                        file_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        indexed_at TIMESTAMP,
                        chunk_count INTEGER DEFAULT 0
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER,
                        chunk_index INTEGER,
                        content TEXT NOT NULL,
                        embedding_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        search_type TEXT,
                        results_count INTEGER,
                        search_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                conn.close()
                
                logger.info(f"Successfully initialized fallback vector database at: {self.vector_db_path}")
                
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback database: {fallback_error}")
                raise RuntimeError(f"Cannot initialize vector database at {self.vector_db_path} or fallback location: {e}")
            
    def _load_model(self):
        """Load the sentence transformer model"""
        if not HAS_VECTOR_DEPS:
            raise ImportError("Vector search dependencies not installed. Please install: pip install sentence-transformers faiss-cpu numpy")
            
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                
                # Try to load the model with error handling
                self.model = SentenceTransformer(self.model_name)
                
                # Test the model with a simple encoding
                test_embedding = self.model.encode(["test"])
                logger.info(f"Embedding model loaded successfully. Test embedding shape: {test_embedding.shape}")
                
            except Exception as e:
                logger.error(f"Error loading embedding model {self.model_name}: {e}")
                
                # Try fallback model
                try:
                    fallback_model = "all-MiniLM-L6-v2"
                    if self.model_name != fallback_model:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        self.model = SentenceTransformer(fallback_model)
                        test_embedding = self.model.encode(["test"])
                        logger.info(f"Fallback model loaded successfully. Test embedding shape: {test_embedding.shape}")
                    else:
                        raise e
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    raise e
                
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
            
    def _extract_text_content(self, file_path: str) -> str:
        """Extract intelligent sample content from files focusing on structure and metadata"""
        file_path_str = str(file_path)
        
        # Handle virtual dataset documents
        if file_path_str.startswith("dataset://"):
            return file_path_str  # Return as-is for virtual documents
        
        file_path = Path(file_path)
        
        try:
            # Check if file exists (skip for virtual documents)
            if not file_path.exists():
                return f"Virtual document: {file_path_str}"
            
            # Check file size first - skip files larger than 50MB
            file_size = file_path.stat().st_size
            max_file_size = 50 * 1024 * 1024  # 50MB
            
            if file_size > max_file_size:
                logger.warning(f"Skipping large file {file_path}: {file_size} bytes > {max_file_size} bytes")
                return self._create_file_summary(file_path, "File too large for content sampling")
            
            # Extract intelligent samples based on file type
            return self._extract_intelligent_sample(file_path)
                    
        except Exception as e:
            logger.warning(f"Could not extract content from {file_path}: {e}")
            return self._create_file_summary(file_path, f"Content extraction failed: {str(e)}")
    
    def _create_file_summary(self, file_path: Path, note: str = "") -> str:
        """Create a structured summary of file metadata without full content"""
        try:
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            summary_parts = [
                f"FILE: {file_path.name}",
                f"PATH: {file_path.parent}",
                f"TYPE: {file_path.suffix.upper()} file",
                f"SIZE: {size_mb:.2f} MB ({stat.st_size:,} bytes)",
                f"MODIFIED: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            if note:
                summary_parts.append(f"NOTE: {note}")
                
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"FILE: {file_path.name}\nERROR: {str(e)}"
    
    def _extract_intelligent_sample(self, file_path: Path) -> str:
        """Extract intelligent samples focusing on structure, schema, and key content"""
        file_ext = file_path.suffix.lower()
        max_sample_size = 8192  # 8KB sample size
        
        try:
            if file_ext == '.csv':
                return self._sample_csv_file(file_path, max_sample_size)
            elif file_ext == '.json':
                return self._sample_json_file(file_path, max_sample_size)
            elif file_ext == '.py':
                return self._sample_python_file(file_path, max_sample_size)
            elif file_ext in ['.sql']:
                return self._sample_sql_file(file_path, max_sample_size)
            elif file_ext in ['.txt', '.md']:
                return self._sample_text_file(file_path, max_sample_size)
            elif file_ext in ['.xlsx', '.xls']:
                return self._sample_excel_file(file_path)
            else:
                # Generic text sampling
                return self._sample_generic_file(file_path, max_sample_size)
                
        except Exception as e:
            logger.warning(f"Error sampling {file_path}: {e}")
            return self._create_file_summary(file_path, f"Sampling failed: {str(e)}")
    
    def _sample_csv_file(self, file_path: Path, max_size: int) -> str:
        """Sample CSV file focusing on headers and data structure"""
        try:
            import csv
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few lines to detect structure
                sample_lines = []
                total_size = 0
                line_count = 0
                
                for line in f:
                    if total_size >= max_size or line_count >= 100:  # Max 100 rows
                        break
                    sample_lines.append(line.strip())
                    total_size += len(line.encode('utf-8'))
                    line_count += 1
                
                if not sample_lines:
                    return self._create_file_summary(file_path, "Empty CSV file")
                
                # Try to detect CSV structure
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample_lines[0]).delimiter
                
                # Parse headers and sample data
                reader = csv.reader(sample_lines, delimiter=delimiter)
                rows = list(reader)
                
                if not rows:
                    return self._create_file_summary(file_path, "No readable CSV data")
                
                headers = rows[0] if rows else []
                sample_data = rows[1:min(6, len(rows))]  # First 5 data rows
                
                # Create structured summary with enhanced column information
                summary_parts = [
                    f"CSV FILE: {file_path.name}",
                    f"COLUMNS ({len(headers)}): {', '.join(headers[:10])}{'...' if len(headers) > 10 else ''}",
                    f"SAMPLE ROWS: {len(sample_data)} of estimated {line_count}+ total rows",
                    f"DELIMITER: '{delimiter}'",
                    ""
                ]
                
                # Add detailed column analysis for better searchability
                if headers:
                    summary_parts.append("COLUMN DETAILS:")
                    for i, header in enumerate(headers[:10]):  # Analyze first 10 columns
                        col_info = f"  {header}"
                        
                        # Analyze sample data for this column if available
                        if sample_data and len(sample_data) > 0:
                            col_values = []
                            for row in sample_data[:3]:
                                if i < len(row) and row[i]:
                                    col_values.append(str(row[i])[:15])
                            
                            if col_values:
                                col_info += f" (examples: {', '.join(col_values)})"
                        
                        summary_parts.append(col_info)
                    
                    if len(headers) > 10:
                        summary_parts.append(f"  ... and {len(headers) - 10} more columns")
                
                # Add sample data
                if sample_data:
                    summary_parts.append("")
                    summary_parts.append("SAMPLE DATA:")
                    for i, row in enumerate(sample_data[:3]):  # Show first 3 rows
                        row_preview = [str(cell)[:20] + "..." if len(str(cell)) > 20 else str(cell) for cell in row[:5]]
                        summary_parts.append(f"Row {i+1}: {' | '.join(row_preview)}")
                
                return "\n".join(summary_parts)
                
        except Exception as e:
            return self._create_file_summary(file_path, f"CSV parsing error: {str(e)}")
    
    def _sample_json_file(self, file_path: Path, max_size: int) -> str:
        """Sample JSON file focusing on structure and schema"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read limited content
                content = f.read(max_size)
                
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # If truncated, try to read smaller amount
                    f.seek(0)
                    content = f.read(max_size // 2)
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        return self._create_file_summary(file_path, "Invalid or truncated JSON")
                
                # Analyze JSON structure
                summary_parts = [f"JSON FILE: {file_path.name}"]
                
                if isinstance(data, dict):
                    keys = list(data.keys())
                    summary_parts.extend([
                        f"TYPE: Object with {len(keys)} keys",
                        f"KEYS: {', '.join(keys[:10])}{'...' if len(keys) > 10 else ''}",
                        ""
                    ])
                    
                    # Sample key-value pairs
                    summary_parts.append("SAMPLE STRUCTURE:")
                    for key in keys[:5]:
                        value = data[key]
                        value_type = type(value).__name__
                        if isinstance(value, (list, dict)):
                            value_preview = f"{value_type} with {len(value)} items"
                        else:
                            value_str = str(value)
                            value_preview = value_str[:50] + "..." if len(value_str) > 50 else value_str
                        summary_parts.append(f"  {key}: {value_preview} ({value_type})")
                        
                elif isinstance(data, list):
                    summary_parts.extend([
                        f"TYPE: Array with {len(data)} items",
                        ""
                    ])
                    
                    if data and isinstance(data[0], dict):
                        # Array of objects - show schema
                        first_item = data[0]
                        keys = list(first_item.keys())
                        summary_parts.extend([
                            f"ITEM SCHEMA: Objects with {len(keys)} keys",
                            f"ITEM KEYS: {', '.join(keys[:10])}{'...' if len(keys) > 10 else ''}",
                            "",
                            "SAMPLE ITEMS:"
                        ])
                        
                        for i, item in enumerate(data[:3]):
                            if isinstance(item, dict):
                                item_preview = {k: str(v)[:30] + "..." if len(str(v)) > 30 else v 
                                              for k, v in list(item.items())[:3]}
                                summary_parts.append(f"  Item {i+1}: {item_preview}")
                    else:
                        # Array of primitives
                        sample_items = [str(item)[:30] + "..." if len(str(item)) > 30 else str(item) 
                                      for item in data[:5]]
                        summary_parts.extend([
                            "SAMPLE VALUES:",
                            f"  {', '.join(sample_items)}"
                        ])
                else:
                    summary_parts.append(f"TYPE: {type(data).__name__} - {str(data)[:100]}")
                
                return "\n".join(summary_parts)
                
        except Exception as e:
            return self._create_file_summary(file_path, f"JSON analysis error: {str(e)}")
    
    def _sample_python_file(self, file_path: Path, max_size: int) -> str:
        """Sample Python file focusing on structure and key elements"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                total_size = 0
                
                for line in f:
                    if total_size >= max_size:
                        break
                    lines.append(line.rstrip())
                    total_size += len(line.encode('utf-8'))
                
                # Analyze Python structure
                imports = []
                classes = []
                functions = []
                docstring = None
                
                in_docstring = False
                docstring_lines = []
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Extract module docstring
                    if i < 10 and ('"""' in stripped or "'''" in stripped) and not docstring:
                        if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                            # Single line docstring
                            docstring = stripped.strip('"""').strip("'''").strip()
                        else:
                            # Multi-line docstring start
                            in_docstring = True
                            docstring_lines = [stripped.strip('"""').strip("'''")]
                    elif in_docstring:
                        if '"""' in stripped or "'''" in stripped:
                            docstring_lines.append(stripped.strip('"""').strip("'''"))
                            docstring = ' '.join(docstring_lines).strip()
                            in_docstring = False
                        else:
                            docstring_lines.append(stripped)
                    
                    # Extract imports
                    elif stripped.startswith(('import ', 'from ')):
                        imports.append(stripped)
                    
                    # Extract class definitions
                    elif stripped.startswith('class '):
                        class_name = stripped.split('(')[0].replace('class ', '').strip(':')
                        classes.append(class_name)
                    
                    # Extract function definitions
                    elif stripped.startswith('def '):
                        func_name = stripped.split('(')[0].replace('def ', '')
                        functions.append(func_name)
                
                # Create summary
                summary_parts = [
                    f"PYTHON FILE: {file_path.name}",
                    f"LINES: {len(lines)} (sampled)"
                ]
                
                if docstring:
                    summary_parts.extend(["", f"DESCRIPTION: {docstring[:200]}{'...' if len(docstring) > 200 else ''}"])
                
                if imports:
                    summary_parts.extend(["", f"IMPORTS ({len(imports)}):"])
                    for imp in imports[:10]:
                        summary_parts.append(f"  {imp}")
                    if len(imports) > 10:
                        summary_parts.append(f"  ... and {len(imports) - 10} more")
                
                if classes:
                    summary_parts.extend(["", f"CLASSES ({len(classes)}): {', '.join(classes[:10])}"])
                
                if functions:
                    summary_parts.extend(["", f"FUNCTIONS ({len(functions)}): {', '.join(functions[:10])}"])
                
                return "\n".join(summary_parts)
                
        except Exception as e:
            return self._create_file_summary(file_path, f"Python analysis error: {str(e)}")
    
    def _sample_sql_file(self, file_path: Path, max_size: int) -> str:
        """Sample SQL file focusing on structure and key statements"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_size)
                
                # Extract SQL statements and structure
                lines = content.split('\n')
                statements = []
                tables = set()
                
                current_statement = []
                
                for line in lines:
                    stripped = line.strip().upper()
                    
                    if any(stripped.startswith(keyword) for keyword in 
                           ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']):
                        if current_statement:
                            statements.append(' '.join(current_statement))
                        current_statement = [line.strip()]
                    elif current_statement:
                        current_statement.append(line.strip())
                        
                    # Extract table names
                    if 'FROM ' in stripped or 'TABLE ' in stripped:
                        words = stripped.split()
                        for i, word in enumerate(words):
                            if word in ['FROM', 'TABLE', 'INTO', 'UPDATE'] and i + 1 < len(words):
                                table_name = words[i + 1].strip('(),;')
                                if table_name and not table_name.upper() in ['SELECT', 'WHERE', 'GROUP']:
                                    tables.add(table_name)
                
                if current_statement:
                    statements.append(' '.join(current_statement))
                
                # Create summary
                summary_parts = [
                    f"SQL FILE: {file_path.name}",
                    f"STATEMENTS: {len(statements)} SQL statements found"
                ]
                
                if tables:
                    summary_parts.append(f"TABLES: {', '.join(list(tables)[:10])}")
                
                if statements:
                    summary_parts.extend(["", "SAMPLE STATEMENTS:"])
                    for i, stmt in enumerate(statements[:3]):
                        stmt_preview = stmt[:100] + "..." if len(stmt) > 100 else stmt
                        summary_parts.append(f"  {i+1}. {stmt_preview}")
                
                return "\n".join(summary_parts)
                
        except Exception as e:
            return self._create_file_summary(file_path, f"SQL analysis error: {str(e)}")
    
    def _sample_text_file(self, file_path: Path, max_size: int) -> str:
        """Sample text/markdown file focusing on structure and key content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_size)
                
                lines = content.split('\n')
                
                # Extract structure for markdown
                if file_path.suffix.lower() == '.md':
                    headers = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('#'):
                            level = len(stripped) - len(stripped.lstrip('#'))
                            header_text = stripped.lstrip('#').strip()
                            headers.append(f"{'  ' * (level-1)}H{level}: {header_text}")
                    
                    summary_parts = [
                        f"MARKDOWN FILE: {file_path.name}",
                        f"LINES: {len(lines)}"
                    ]
                    
                    if headers:
                        summary_parts.extend(["", "STRUCTURE:"] + headers[:15])
                    
                    # Add content preview
                    content_preview = content[:500] + "..." if len(content) > 500 else content
                    summary_parts.extend(["", "CONTENT PREVIEW:", content_preview])
                    
                else:
                    # Regular text file
                    summary_parts = [
                        f"TEXT FILE: {file_path.name}",
                        f"LINES: {len(lines)}",
                        f"CHARACTERS: {len(content)}",
                        "",
                        "CONTENT PREVIEW:",
                        content[:500] + "..." if len(content) > 500 else content
                    ]
                
                return "\n".join(summary_parts)
                
        except Exception as e:
            return self._create_file_summary(file_path, f"Text analysis error: {str(e)}")
    
    def _sample_excel_file(self, file_path: Path) -> str:
        """Sample Excel file (metadata only since we don't have pandas)"""
        return self._create_file_summary(file_path, "Excel file - install pandas/openpyxl for content analysis")
    
    def _sample_generic_file(self, file_path: Path, max_size: int) -> str:
        """Generic file sampling for unknown types"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(min(max_size, 1024))  # Smaller sample for unknown types
                
                summary_parts = [
                    f"FILE: {file_path.name}",
                    f"TYPE: {file_path.suffix.upper() or 'Unknown'} file",
                    "",
                    "CONTENT SAMPLE:",
                    content[:300] + "..." if len(content) > 300 else content
                ]
                
                return "\n".join(summary_parts)
                
        except Exception as e:
            return self._create_file_summary(file_path, f"Generic sampling error: {str(e)}")
    
    def _extract_content_metadata(self, content: str, file_type: str) -> Dict[str, Any]:
        """Extract additional metadata from content for enhanced indexing"""
        metadata = {
            'tags': [],
            'description': '',
            'columns': [],
            'keywords': []
        }
        
        try:
            if file_type == '.csv':
                # Extract CSV column information for better searchability
                lines = content.split('\n')
                if lines and 'COLUMNS' in content:
                    # Find column information in the content
                    for line in lines:
                        if line.startswith('COLUMNS'):
                            # Extract column names
                            columns_part = line.split(':', 1)[1] if ':' in line else line
                            # Remove count information and clean up
                            columns_part = columns_part.split('(')[0] if '(' in columns_part else columns_part
                            columns = [col.strip() for col in columns_part.split(',')]
                            metadata['columns'] = columns
                            
                            # Add column names as searchable keywords
                            metadata['keywords'].extend(columns)
                            
                            # Generate tags based on column names
                            for col in columns:
                                col_lower = col.lower()
                                if any(keyword in col_lower for keyword in ['income', 'salary', 'wage', 'pay']):
                                    metadata['tags'].append('financial')
                                if any(keyword in col_lower for keyword in ['name', 'customer', 'user', 'person']):
                                    metadata['tags'].append('personal')
                                if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated']):
                                    metadata['tags'].append('temporal')
                                if any(keyword in col_lower for keyword in ['id', 'key', 'index']):
                                    metadata['tags'].append('identifier')
                                if any(keyword in col_lower for keyword in ['address', 'location', 'city', 'state']):
                                    metadata['tags'].append('geographic')
                            
                            # Create description based on columns
                            if len(columns) > 0:
                                metadata['description'] = f"CSV file with {len(columns)} columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}"
                            break
                            
            elif file_type == '.json':
                # Extract JSON structure information
                if 'KEYS:' in content:
                    for line in content.split('\n'):
                        if line.strip().startswith('KEYS:'):
                            keys_part = line.split(':', 1)[1] if ':' in line else line
                            keys = [key.strip() for key in keys_part.split(',')]
                            metadata['keywords'].extend(keys)
                            
                            # Generate tags based on JSON keys
                            for key in keys:
                                key_lower = key.lower()
                                if any(keyword in key_lower for keyword in ['config', 'setting', 'option']):
                                    metadata['tags'].append('configuration')
                                if any(keyword in key_lower for keyword in ['data', 'record', 'item']):
                                    metadata['tags'].append('data')
                                if any(keyword in key_lower for keyword in ['api', 'endpoint', 'service']):
                                    metadata['tags'].append('api')
                            break
                            
            elif file_type == '.py':
                # Extract Python code information
                if 'CLASSES' in content:
                    for line in content.split('\n'):
                        if 'CLASSES' in line and ':' in line:
                            classes_part = line.split(':', 1)[1]
                            classes = [cls.strip() for cls in classes_part.split(',')]
                            metadata['keywords'].extend(classes)
                            metadata['tags'].append('object-oriented')
                            break
                            
                if 'FUNCTIONS' in content:
                    for line in content.split('\n'):
                        if 'FUNCTIONS' in line and ':' in line:
                            functions_part = line.split(':', 1)[1]
                            functions = [func.strip() for func in functions_part.split(',')]
                            metadata['keywords'].extend(functions[:10])  # Limit to first 10
                            metadata['tags'].append('functional')
                            break
                            
            # Extract general keywords from content
            content_lower = content.lower()
            
            # Look for common data-related terms
            data_keywords = ['income', 'revenue', 'sales', 'profit', 'customer', 'user', 'transaction', 
                           'order', 'product', 'service', 'account', 'balance', 'payment', 'invoice']
            
            for keyword in data_keywords:
                if keyword in content_lower:
                    metadata['keywords'].append(keyword)
                    
            # Remove duplicates from tags and keywords
            metadata['tags'] = list(set(metadata['tags']))
            metadata['keywords'] = list(set(metadata['keywords']))
            
        except Exception as e:
            logger.warning(f"Error extracting content metadata: {e}")
            
        return metadata
        
    def _chunk_text(self, text: str) -> List[str]:
        """Split structured content into logical chunks focusing on sections and metadata"""
        # For structured content samples, we typically want fewer, more meaningful chunks
        max_chunk_size = min(self.max_chunk_size, 2048)  # Smaller chunks for samples
        
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        
        # Try to split on logical boundaries first
        logical_separators = [
            '\n---\n',  # Our content separator
            '\nSAMPLE DATA:\n',
            '\nSTRUCTURE:\n', 
            '\nCONTENT PREVIEW:\n',
            '\n\n',  # Paragraph breaks
            '\n'     # Line breaks
        ]
        
        # Split on the first separator that creates reasonable chunks
        for separator in logical_separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                
                for part in parts:
                    # If adding this part would exceed chunk size, save current chunk
                    if len(current_chunk) + len(part) + len(separator) > max_chunk_size:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If we got reasonable chunks, return them
                if len(chunks) > 1 and all(len(chunk) < max_chunk_size for chunk in chunks):
                    return chunks
                else:
                    chunks = []  # Reset and try next separator
        
        # Fallback to character-based chunking if logical splitting didn't work
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence or line boundaries
            if end < len(text):
                for i in range(end, max(start + max_chunk_size - 200, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
                        
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end - self.chunk_overlap
            if start >= len(text):
                break
                
        return chunks if chunks else [text]  # Always return at least one chunk
        
    def _create_enhanced_content(self, chunk: str, title: str, fileset_name: str = None, 
                               fileset_description: str = None, tags: List[str] = None,
                               user_description: str = None, schema_info: str = None) -> str:
        """Create enhanced content for vector search prioritizing metadata and structure over raw content"""
        enhanced_parts = []
        
        # Prioritize structured metadata for better search context
        if fileset_name:
            enhanced_parts.append(f"DATASET: {fileset_name}")
        if title:
            enhanced_parts.append(f"FILE: {title}")
        if fileset_description:
            enhanced_parts.append(f"PURPOSE: {fileset_description}")
        if user_description:
            enhanced_parts.append(f"DESCRIPTION: {user_description}")
        if schema_info:
            enhanced_parts.append(f"SCHEMA: {schema_info}")
        if tags:
            enhanced_parts.append(f"TAGS: {', '.join(tags)}")
            
        # Generate searchable keywords from metadata
        search_keywords = set()
        if fileset_name:
            # Split on common separators and add variations
            for word in fileset_name.lower().replace('_', ' ').replace('-', ' ').split():
                search_keywords.add(word)
        if tags:
            for tag in tags:
                search_keywords.add(tag.lower())
        if title:
            for word in title.lower().replace('_', ' ').replace('-', ' ').split():
                search_keywords.add(word)
                
        if search_keywords:
            enhanced_parts.append(f"KEYWORDS: {' '.join(sorted(search_keywords))}")
            
        # Add file path context for better discovery
        if title and '/' in chunk:  # Likely contains path info
            enhanced_parts.append(f"LOCATION: {title}")
            
        # Add the structured content sample (not full content)
        enhanced_parts.append("---")
        enhanced_parts.append("CONTENT STRUCTURE:")
        enhanced_parts.append(chunk)
        
        return '\n'.join(enhanced_parts)
        
    def index_document(self, file_path: str, fileset_name: str = None, fileset_description: str = None, 
                      schema_info: str = None, tags: List[str] = None, user_description: str = None) -> bool:
        """Index a single document with enhanced metadata and duplicate prevention using fully qualified paths"""
        try:
            # Create fully qualified unique key to prevent duplicates
            if file_path.startswith("dataset://"):
                # For datasets, use the full dataset URI as unique key
                unique_key = file_path
                file_hash = hashlib.md5(file_path.encode()).hexdigest()
                content = user_description or f"Virtual dataset document: {file_path}"
                file_size = len(content.encode('utf-8'))
                file_type = "dataset"
                title = file_path.replace("dataset://", "")
            else:
                # For files, always use absolute path as unique key
                file_path_obj = Path(file_path)
                if not file_path_obj.is_absolute():
                    file_path_obj = file_path_obj.resolve()
                unique_key = str(file_path_obj).replace('\\', '/')  # Normalize path separators
                
                file_hash = self._get_file_hash(str(file_path_obj))
                
                # Extract text content with enhanced metadata extraction
                content = self._extract_text_content(str(file_path_obj))
                if not content.strip():
                    logger.warning(f"No content extracted from {unique_key}")
                    return False
                    
                # Get file metadata
                file_stat = file_path_obj.stat()
                file_size = file_stat.st_size
                file_type = file_path_obj.suffix.lower()
                title = file_path_obj.stem
            
            # Check database exists
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                logger.error(f"Vector database not found at: {db_path_str}")
                return False
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            # Check for existing document using unique key for comprehensive duplicate detection
            cursor.execute("""
                SELECT id, file_hash, indexed_at, fileset_name, fileset_description, 
                       schema_info, tags, user_description, chunk_count
                FROM documents WHERE file_path = ?
            """, (unique_key,))
            existing_doc = cursor.fetchone()
            
            # Determine if we need to update or create new
            needs_content_reindex = False
            needs_metadata_update = False
            document_id = None
            
            if existing_doc:
                document_id = existing_doc[0]
                existing_hash = existing_doc[1]
                existing_indexed_at = existing_doc[2]
                existing_fileset = existing_doc[3]
                existing_fileset_desc = existing_doc[4]
                existing_schema = existing_doc[5]
                existing_tags = existing_doc[6]
                existing_user_desc = existing_doc[7]
                existing_chunk_count = existing_doc[8]
                
                # Check if content changed (needs re-indexing)
                if existing_hash != file_hash or not existing_indexed_at:
                    needs_content_reindex = True
                    logger.info(f"Content changed for {unique_key}, will re-index")
                
                # Check if metadata changed (needs metadata merge)
                if (fileset_name and fileset_name != existing_fileset) or \
                   (fileset_description and fileset_description != existing_fileset_desc) or \
                   (schema_info and schema_info != existing_schema) or \
                   (tags and ','.join(tags) != existing_tags) or \
                   (user_description and user_description != existing_user_desc):
                    needs_metadata_update = True
                    logger.info(f"Metadata changed for {unique_key}, will merge")
                
                # If no changes needed, return success
                if not needs_content_reindex and not needs_metadata_update:
                    logger.debug(f"Document {unique_key} already up-to-date")
                    conn.close()
                    return True
            else:
                # New document
                needs_content_reindex = True
                needs_metadata_update = True
                logger.info(f"New document to index: {unique_key}")
            
            # Prepare metadata with intelligent merging
            content_preview = content[:200] + "..." if len(content) > 200 else content
            
            # Detect schema for structured files with enhanced metadata extraction
            detected_schema = detect_schema(unique_key, content)
            
            # Extract additional metadata from content for better indexing
            extracted_metadata = self._extract_content_metadata(content, file_type)
            
            # Merge metadata intelligently
            if existing_doc and needs_metadata_update:
                # Merge metadata - prefer new values but keep existing if new is empty
                final_fileset = fileset_name or existing_doc[3] or generate_fileset_name(unique_key)
                final_fileset_desc = fileset_description or existing_doc[4]
                final_schema = schema_info or existing_doc[5] or detected_schema
                
                # Merge tags - combine existing, new, and extracted tags
                existing_tags_list = existing_doc[6].split(',') if existing_doc[6] else []
                new_tags_list = tags if tags else []
                extracted_tags_list = extracted_metadata.get('tags', [])
                merged_tags = list(set(existing_tags_list + new_tags_list + extracted_tags_list))
                merged_tags = [tag.strip() for tag in merged_tags if tag.strip()]  # Clean empty tags
                final_tags = merged_tags
                
                # Enhance user description with extracted metadata
                base_desc = user_description or existing_doc[7] or ""
                extracted_desc = extracted_metadata.get('description', "")
                if extracted_desc and extracted_desc not in base_desc:
                    final_user_desc = f"{base_desc}\n\n{extracted_desc}".strip()
                else:
                    final_user_desc = base_desc
                
                logger.info(f"Merging metadata for {unique_key}: "
                          f"fileset='{final_fileset}', tags={len(final_tags)}")
            else:
                # New document or no metadata to merge
                final_fileset = fileset_name or generate_fileset_name(unique_key)
                final_fileset_desc = fileset_description
                final_schema = schema_info or detected_schema
                
                # Combine provided tags with extracted tags
                provided_tags = tags if tags else []
                extracted_tags = extracted_metadata.get('tags', [])
                final_tags = list(set(provided_tags + extracted_tags))
                
                # Enhance user description with extracted metadata
                base_desc = user_description or ""
                extracted_desc = extracted_metadata.get('description', "")
                if extracted_desc and extracted_desc not in base_desc:
                    final_user_desc = f"{base_desc}\n\n{extracted_desc}".strip()
                else:
                    final_user_desc = base_desc
            
            # Convert tags to string
            tags_str = ','.join(final_tags) if final_tags else ''
            
            # Update or create fileset record (avoid duplicates)
            if final_fileset:
                cursor.execute("""
                    INSERT OR REPLACE INTO filesets 
                    (name, description, schema_info, tags, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (final_fileset, final_fileset_desc, final_schema, tags_str, datetime.now()))
            
            # Handle document record
            if existing_doc:
                # Update existing document
                cursor.execute("""
                    UPDATE documents 
                    SET file_hash = ?, title = ?, content_preview = ?, file_type = ?, file_size = ?,
                        fileset_name = ?, fileset_description = ?, schema_info = ?, tags = ?, 
                        user_description = ?, updated_at = ?
                    WHERE id = ?
                """, (file_hash, title, content_preview, file_type, file_size,
                      final_fileset, final_fileset_desc, final_schema, tags_str, 
                      final_user_desc, datetime.now(), document_id))
                
                logger.info(f"Updated existing document record for {unique_key}")
            else:
                # Insert new document with unique key
                cursor.execute("""
                    INSERT INTO documents 
                    (file_path, file_hash, title, content_preview, file_type, file_size, 
                     fileset_name, fileset_description, schema_info, tags, user_description, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (unique_key, file_hash, title, content_preview, file_type, file_size,
                      final_fileset, final_fileset_desc, final_schema, tags_str, final_user_desc, datetime.now()))
                
                document_id = cursor.lastrowid
                logger.info(f"Created new document record for {unique_key}")
            
            # Handle content re-indexing if needed
            if needs_content_reindex:
                # Delete existing chunks and embeddings for this document
                cursor.execute("SELECT embedding_id FROM chunks WHERE document_id = ?", (document_id,))
                old_embeddings = cursor.fetchall()
                
                # Delete old embedding files
                for (embedding_id,) in old_embeddings:
                    embedding_file = self.embeddings_path / f"{embedding_id}.npy"
                    if embedding_file.exists():
                        try:
                            embedding_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to delete old embedding {embedding_id}: {e}")
                
                # Delete old chunks
                cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                
                # Chunk the content
                chunks = self._chunk_text(content)
                
                # Load model if needed
                self._load_model()
                
                # Generate embeddings and store chunks with enhanced content
                for i, chunk in enumerate(chunks):
                    embedding_id = f"{document_id}_{i}"
                    
                    # Create enhanced content that includes metadata for better search
                    enhanced_content = self._create_enhanced_content(
                        chunk, title, final_fileset, final_fileset_desc, final_tags, 
                        final_user_desc, final_schema
                    )
                    
                    cursor.execute("""
                        INSERT INTO chunks (document_id, chunk_index, content, enhanced_content, embedding_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, (document_id, i, chunk, enhanced_content, embedding_id))
                    
                    # Generate and cache embedding using enhanced content
                    try:
                        embedding = self.model.encode([enhanced_content])[0]
                        embedding_file = self.embeddings_path / f"{embedding_id}.npy"
                        np.save(embedding_file, embedding)
                    except Exception as e:
                        logger.error(f"Error generating embedding for chunk {embedding_id}: {e}")
                        
                # Update document with indexing info
                cursor.execute("""
                    UPDATE documents 
                    SET indexed_at = ?, chunk_count = ?
                    WHERE id = ?
                """, (datetime.now(), len(chunks), document_id))
                
                logger.info(f"Re-indexed content for {unique_key} with {len(chunks)} chunks")
            else:
                # Just update metadata in existing chunks if metadata changed
                if needs_metadata_update and existing_doc:
                    cursor.execute("SELECT id, chunk_index, content FROM chunks WHERE document_id = ?", (document_id,))
                    existing_chunks = cursor.fetchall()
                    
                    for chunk_id, chunk_index, chunk_content in existing_chunks:
                        # Update enhanced content with new metadata
                        enhanced_content = self._create_enhanced_content(
                            chunk_content, title, final_fileset, final_fileset_desc, 
                            final_tags, final_user_desc, final_schema
                        )
                        
                        cursor.execute("""
                            UPDATE chunks SET enhanced_content = ? WHERE id = ?
                        """, (enhanced_content, chunk_id))
                        
                        # Regenerate embedding with updated metadata
                        try:
                            embedding_id = f"{document_id}_{chunk_index}"
                            embedding = self.model.encode([enhanced_content])[0]
                            embedding_file = self.embeddings_path / f"{embedding_id}.npy"
                            np.save(embedding_file, embedding)
                        except Exception as e:
                            logger.error(f"Error updating embedding for chunk {embedding_id}: {e}")
                    
                    logger.info(f"Updated metadata for {len(existing_chunks)} chunks in {unique_key}")
            
            conn.commit()
            conn.close()
            
            action = "re-indexed" if needs_content_reindex else "updated metadata for"
            logger.info(f"Successfully {action} document {unique_key} in fileset '{final_fileset}'")
            return True
            
        except Exception as e:
            tbl = traceback.format_exc()
            logger.error(tbl)
            logger.error(f"Error indexing document {file_path}: {e}")
            return False
            
    def index_directory(self, directory_path: str, file_extensions: List[str] = None, 
                       fileset_name: str = None, fileset_description: str = None,
                       tags: List[str] = None, max_files: int = 1000) -> Dict[str, Any]:
        """Index supported files in a directory with dataset metadata and limits"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.py', '.json', '.csv', '.sql']
            
        directory_path = Path(directory_path)
        
        # Auto-generate fileset name if not provided
        if not fileset_name:
            fileset_name = directory_path.name
            
        results = {
            'total_files': 0,
            'indexed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'errors': [],
            'fileset_name': fileset_name,
            'truncated': False
        }
        
        try:
            # First pass: count and collect files with size filtering
            all_files = []
            max_file_size = 10 * 1024 * 1024  # 10MB limit
            
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                    try:
                        # Check file size before adding to list
                        file_size = file_path.stat().st_size
                        if file_size <= max_file_size:
                            all_files.append(file_path)
                        else:
                            results['skipped_files'] += 1
                            logger.info(f"Skipping large file: {file_path} ({file_size:,} bytes)")
                            
                        # Stop if we've found too many files
                        if len(all_files) >= max_files:
                            results['truncated'] = True
                            logger.warning(f"Limiting indexing to first {max_files} files in {directory_path}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error checking file {file_path}: {e}")
                        results['skipped_files'] += 1
                    
            results['total_files'] = len(all_files)
            
            # Second pass: index files with metadata
            for i, file_path in enumerate(all_files):
                try:
                    # Log progress every 50 files
                    if i % 50 == 0:
                        logger.info(f"Indexing progress: {i}/{len(all_files)} files")
                        
                    if self.index_document(
                        str(file_path), 
                        fileset_name=fileset_name,
                        fileset_description=fileset_description,
                        tags=tags
                    ):
                        results['indexed_files'] += 1
                        logger.debug(f"Successfully indexed: {file_path}")
                    else:
                        results['skipped_files'] += 1
                        logger.debug(f"Skipped (no content): {file_path}")
                        
                except Exception as e:
                    results['failed_files'] += 1
                    error_msg = f"{file_path.name}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(f"Failed to index {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Error indexing directory {directory_path}: {e}")
            results['errors'].append(f"Directory error: {str(e)}")
            
        return results
        
    def get_filesets(self) -> List[Dict[str, Any]]:
        """Get all filesets/datasets"""
        try:
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                return []
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT f.name, f.description, f.schema_info, f.tags, f.created_at,
                       COUNT(d.id) as file_count
                FROM filesets f
                LEFT JOIN documents d ON f.name = d.fileset_name
                GROUP BY f.name, f.description, f.schema_info, f.tags, f.created_at
                ORDER BY f.updated_at DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'name': row[0],
                    'description': row[1],
                    'schema_info': row[2],
                    'tags': row[3].split(',') if row[3] else [],
                    'created_at': row[4],
                    'file_count': row[5]
                })
                
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting filesets: {e}")
            return []
            
    def update_document_metadata(self, file_path: str, fileset_name: str = None, 
                               fileset_description: str = None, tags: List[str] = None,
                               user_description: str = None) -> bool:
        """Update metadata for an existing document using the main indexing method with unique key normalization"""
        try:
            # Normalize file path to match the unique key system
            if file_path.startswith("dataset://"):
                normalized_path = file_path
            else:
                file_path_obj = Path(file_path)
                if not file_path_obj.is_absolute():
                    file_path_obj = file_path_obj.resolve()
                normalized_path = str(file_path_obj).replace('\\', '/')
            
            # Use the main index_document method which now handles metadata merging
            return self.index_document(
                file_path=normalized_path,
                fileset_name=fileset_name,
                fileset_description=fileset_description,
                tags=tags,
                user_description=user_description
            )
            
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            return False
        
    def search(self, query: str, max_results: int = 10, similarity_threshold: float = 0.3, 
               search_mode: str = "hybrid") -> List[Dict[str, Any]]:
        """Perform enhanced vector search with multiple algorithms"""
        if not HAS_VECTOR_DEPS:
            return [{
                'error': 'Vector search dependencies not installed',
                'message': 'Please install: pip install sentence-transformers faiss-cpu numpy'
            }]
            
        try:
            start_time = datetime.now()
            logger.info(f"Starting enhanced vector search for query: '{query}' with mode: {search_mode}")
            
            # Load model if needed
            try:
                self._load_model()
                if self.model is None:
                    return [{
                        'error': 'Model loading failed',
                        'message': 'Could not load the sentence transformer model'
                    }]
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return [{
                    'error': 'Model loading failed',
                    'message': f'Error loading model: {str(e)}'
                }]
            
            # Parse and understand the query
            query_info = self._analyze_query(query)
            logger.debug(f"Query analysis: {query_info}")
            
            # Expand query with synonyms and related terms
            expanded_queries = self._expand_query(query, query_info)
            logger.debug(f"Expanded queries: {expanded_queries}")
            
            # Get database connection
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                logger.error(f"Vector database not found at: {db_path_str}")
                return [{
                    'message': 'Vector database not found',
                    'suggestion': 'Please rebuild the vector search index',
                    'database_path': db_path_str
                }]
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            # Check database status
            cursor.execute("SELECT COUNT(*) FROM documents WHERE indexed_at IS NOT NULL")
            indexed_docs = cursor.fetchone()[0]
            
            if indexed_docs == 0:
                conn.close()
                return [{
                    'message': 'No indexed documents found',
                    'suggestion': 'Please index some documents first'
                }]
            
            # Perform different search strategies based on mode
            if search_mode == "hybrid":
                results = self._hybrid_search(query, expanded_queries, query_info, cursor, max_results, similarity_threshold)
            elif search_mode == "semantic":
                results = self._semantic_search(query, expanded_queries, cursor, max_results, similarity_threshold)
            elif search_mode == "keyword":
                results = self._keyword_search(query, query_info, cursor, max_results)
            else:
                # Default to hybrid
                results = self._hybrid_search(query, expanded_queries, query_info, cursor, max_results, similarity_threshold)
            
            # Re-rank results using multiple signals
            results = self._rerank_results(results, query, query_info)
            
            # Diversify results to avoid too many from same source
            results = self._diversify_results(results, max_results)
            
            # Log search to database
            try:
                search_time = (datetime.now() - start_time).total_seconds()
                cursor.execute("""
                    INSERT INTO search_history (query, search_type, results_count, search_time)
                    VALUES (?, ?, ?, ?)
                """, (query, f'enhanced_{search_mode}', len(results), search_time))
                conn.commit()
            except Exception as e:
                logger.warning(f"Failed to log search history: {e}")
            finally:
                conn.close()
            
            search_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Enhanced vector search returned {len(results)} results in {search_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing enhanced vector search: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [{
                'error': 'Search failed',
                'message': str(e),
                'traceback': traceback.format_exc()
            }]
            
    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Get list of all indexed documents"""
        try:
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                logger.warning(f"Vector database not found at: {db_path_str}")
                return []
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, file_path, title, file_type, file_size, 
                       chunk_count, indexed_at
                FROM documents
                WHERE indexed_at IS NOT NULL
                ORDER BY indexed_at DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'file_path': row[1],
                    'title': row[2],
                    'file_type': row[3],
                    'file_size': row[4],
                    'chunk_count': row[5],
                    'indexed_at': row[6]
                })
                
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting indexed documents: {e}")
            return []
            
    def clear_index(self):
        """Clear all indexed documents and embeddings"""
        try:
            # Clear database
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                logger.warning(f"Vector database not found at: {db_path_str}")
                # Re-initialize the database
                self._init_database()
                return
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            # Clear in proper order to respect foreign key constraints
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM filesets")
            cursor.execute("DELETE FROM search_history")
            
            conn.commit()
            conn.close()
            
            # Clear embedding files
            if self.embeddings_path.exists():
                for embedding_file in self.embeddings_path.glob("*.npy"):
                    try:
                        embedding_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete embedding file {embedding_file}: {e}")
                        
            logger.info("Vector search index cleared completely")
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise
            
    def remove_duplicate_documents(self) -> Dict[str, Any]:
        """Remove duplicate documents based on normalized file paths and merge their metadata"""
        try:
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                return {'removed': 0, 'errors': []}
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            # Find potential duplicates by normalizing paths
            cursor.execute("SELECT id, file_path FROM documents")
            all_docs = cursor.fetchall()
            
            # Group documents by normalized paths
            path_groups = {}
            for doc_id, file_path in all_docs:
                # Normalize the path for comparison
                if file_path.startswith("dataset://"):
                    normalized = file_path
                else:
                    try:
                        normalized = str(Path(file_path).resolve()).replace('\\', '/')
                    except:
                        normalized = file_path.replace('\\', '/')
                
                if normalized not in path_groups:
                    path_groups[normalized] = []
                path_groups[normalized].append((doc_id, file_path))
            
            removed_count = 0
            errors = []
            
            # Process groups with duplicates
            for normalized_path, docs in path_groups.items():
                if len(docs) <= 1:
                    continue  # No duplicates
                
                try:
                    doc_ids = [doc[0] for doc in docs]
                    
                    # Get all duplicate records
                    cursor.execute("""
                        SELECT id, file_path, fileset_name, fileset_description, schema_info, tags, 
                               user_description, indexed_at, chunk_count, updated_at
                        FROM documents WHERE id IN ({})
                        ORDER BY updated_at DESC, indexed_at DESC
                    """.format(','.join('?' * len(doc_ids))), doc_ids)
                    
                    records = cursor.fetchall()
                    
                    # Keep the most recently updated record (first in list)
                    keep_record = records[0]
                    keep_id = keep_record[0]
                    
                    # Merge metadata from all records
                    merged_fileset = None
                    merged_description = None
                    merged_schema = None
                    merged_tags = set()
                    merged_user_desc_parts = []
                    
                    for record in records:
                        if record[2] and not merged_fileset:  # fileset_name
                            merged_fileset = record[2]
                        if record[3] and not merged_description:  # fileset_description
                            merged_description = record[3]
                        if record[4] and not merged_schema:  # schema_info
                            merged_schema = record[4]
                        if record[5]:  # tags
                            merged_tags.update([tag.strip() for tag in record[5].split(',') if tag.strip()])
                        if record[6]:  # user_description
                            desc = record[6].strip()
                            if desc and desc not in merged_user_desc_parts:
                                merged_user_desc_parts.append(desc)
                    
                    # Combine user descriptions
                    merged_user_desc = '\n\n'.join(merged_user_desc_parts) if merged_user_desc_parts else None
                    
                    # Clean and merge tags
                    merged_tags_str = ','.join(sorted(merged_tags)) if merged_tags else ''
                    
                    # Update the kept record with merged metadata and normalized path
                    cursor.execute("""
                        UPDATE documents 
                        SET file_path = ?, fileset_name = ?, fileset_description = ?, schema_info = ?, 
                            tags = ?, user_description = ?, updated_at = ?
                        WHERE id = ?
                    """, (normalized_path, merged_fileset, merged_description, merged_schema, 
                          merged_tags_str, merged_user_desc, datetime.now(), keep_id))
                    
                    # Remove duplicate records (except the one we're keeping)
                    remove_ids = [record[0] for record in records[1:]]
                    
                    for remove_id in remove_ids:
                        # Delete chunks and embeddings for duplicate records
                        cursor.execute("SELECT embedding_id FROM chunks WHERE document_id = ?", (remove_id,))
                        embeddings = cursor.fetchall()
                        
                        for (embedding_id,) in embeddings:
                            embedding_file = self.embeddings_path / f"{embedding_id}.npy"
                            if embedding_file.exists():
                                try:
                                    embedding_file.unlink()
                                except Exception as e:
                                    logger.warning(f"Failed to delete embedding {embedding_id}: {e}")
                        
                        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (remove_id,))
                        cursor.execute("DELETE FROM documents WHERE id = ?", (remove_id,))
                        
                        removed_count += 1
                    
                    logger.info(f"Merged {len(docs)} duplicates for {normalized_path}, kept record {keep_id}")
                    
                except Exception as e:
                    error_msg = f"Error processing duplicates for {normalized_path}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Removed {removed_count} duplicate documents")
            return {'removed': removed_count, 'errors': errors}
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            return {'removed': 0, 'errors': [str(e)]}
            
    def rebuild_index(self, directories: List[str]) -> Dict[str, Any]:
        """Rebuild the entire vector search index"""
        try:
            # Clear existing index
            self.clear_index()
            
            # Index all directories
            total_results = {
                'total_files': 0,
                'indexed_files': 0,
                'failed_files': 0,
                'skipped_files': 0,
                'errors': []
            }
            
            for directory in directories:
                if not Path(directory).exists():
                    total_results['errors'].append(f"Directory not found: {directory}")
                    continue
                    
                results = self.index_directory(directory)
                
                total_results['total_files'] += results['total_files']
                total_results['indexed_files'] += results['indexed_files']
                total_results['failed_files'] += results['failed_files']
                total_results['skipped_files'] += results['skipped_files']
                total_results['errors'].extend(results['errors'])
                
            logger.info(f"Index rebuild complete: {total_results['indexed_files']}/{total_results['total_files']} files indexed")
            return total_results
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise
            
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                return {
                    'document_count': 0,
                    'chunk_count': 0,
                    'total_size_bytes': 0,
                    'file_types': {},
                    'recent_indexing_count': 0,
                    'embeddings_path': str(self.embeddings_path),
                    'database_path': str(self.vector_db_path),
                    'status': 'Database not found',
                    'is_empty': True
                }
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            # Get document count
            cursor.execute("SELECT COUNT(*) FROM documents WHERE indexed_at IS NOT NULL")
            doc_count = cursor.fetchone()[0]
            
            # Get chunk count
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Get total file size
            cursor.execute("SELECT SUM(file_size) FROM documents WHERE indexed_at IS NOT NULL")
            total_size = cursor.fetchone()[0] or 0
            
            # Get file types
            cursor.execute("""
                SELECT file_type, COUNT(*) 
                FROM documents 
                WHERE indexed_at IS NOT NULL 
                GROUP BY file_type
            """)
            file_types = dict(cursor.fetchall())
            
            # Get recent indexing activity
            cursor.execute("""
                SELECT COUNT(*) 
                FROM documents 
                WHERE indexed_at > datetime('now', '-24 hours')
            """)
            recent_count = cursor.fetchone()[0]
            
            # Get fileset count
            cursor.execute("SELECT COUNT(*) FROM filesets")
            fileset_count = cursor.fetchone()[0]
            
            # Get top tags
            cursor.execute("""
                SELECT tags, COUNT(*) as count
                FROM documents 
                WHERE tags IS NOT NULL AND tags != ''
                GROUP BY tags
                ORDER BY count DESC
                LIMIT 10
            """)
            top_tags = dict(cursor.fetchall())
            
            # Count embedding files
            embedding_count = 0
            if self.embeddings_path.exists():
                embedding_count = len(list(self.embeddings_path.glob("*.npy")))
            
            # Get database file size
            db_size = Path(db_path_str).stat().st_size if Path(db_path_str).exists() else 0
            
            conn.close()
            
            # Determine status
            if doc_count > 0:
                status = f'Ready - {doc_count} documents indexed'
            else:
                status = 'Empty database - no documents indexed'
            
            return {
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'embedding_count': embedding_count,
                'fileset_count': fileset_count,
                'total_size_bytes': total_size,
                'database_size_bytes': db_size,
                'file_types': file_types,
                'recent_indexing_count': recent_count,
                'top_tags': top_tags,
                'embeddings_path': str(self.embeddings_path),
                'database_path': str(self.vector_db_path),
                'status': status,
                'is_empty': doc_count == 0
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                'status': f'Error: {str(e)}',
                'embeddings_path': str(self.embeddings_path),
                'database_path': str(self.vector_db_path),
                'is_empty': True
            }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and extract key information"""
        query_lower = query.lower()
        
        analysis = {
            'intent': 'general',
            'entities': [],
            'keywords': [],
            'filters': {},
            'question_type': None,
            'data_types': [],
            'operations': []
        }
        
        # Detect question types
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        for word in question_words:
            if query_lower.startswith(word):
                analysis['question_type'] = word
                analysis['intent'] = 'question'
                break
        
        # Detect data-related terms
        data_terms = {
            'financial': ['income', 'revenue', 'profit', 'sales', 'money', 'cost', 'budget', 'financial'],
            'customer': ['customer', 'client', 'user', 'person', 'people', 'demographic'],
            'temporal': ['date', 'time', 'year', 'month', 'day', 'recent', 'latest', 'historical'],
            'analytics': ['analysis', 'report', 'dashboard', 'metric', 'kpi', 'trend', 'pattern'],
            'technical': ['code', 'function', 'class', 'api', 'database', 'table', 'column']
        }
        
        for category, terms in data_terms.items():
            if any(term in query_lower for term in terms):
                analysis['data_types'].append(category)
        
        # Detect operations
        operations = {
            'aggregation': ['sum', 'count', 'average', 'total', 'maximum', 'minimum'],
            'filtering': ['filter', 'where', 'contains', 'includes', 'exclude'],
            'comparison': ['compare', 'versus', 'vs', 'difference', 'similar', 'like'],
            'trend': ['trend', 'change', 'growth', 'decline', 'increase', 'decrease']
        }
        
        for op_type, terms in operations.items():
            if any(term in query_lower for term in terms):
                analysis['operations'].append(op_type)
        
        # Extract potential entities (simple approach)
        words = query.split()
        analysis['keywords'] = [word.strip('.,!?') for word in words if len(word) > 2]
        
        return analysis
    
    def _expand_query(self, query: str, query_info: Dict[str, Any]) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded = [query]  # Always include original query
        
        # Synonym mapping for common data terms
        synonyms = {
            'income': ['revenue', 'earnings', 'salary', 'wages', 'profit'],
            'customer': ['client', 'user', 'buyer', 'consumer'],
            'sales': ['revenue', 'transactions', 'purchases', 'orders'],
            'data': ['information', 'records', 'dataset', 'table'],
            'analysis': ['analytics', 'report', 'insights', 'metrics'],
            'financial': ['monetary', 'fiscal', 'economic', 'budget'],
            'performance': ['metrics', 'kpi', 'results', 'outcomes']
        }
        
        query_lower = query.lower()
        
        # Add synonyms for detected terms
        for term, syns in synonyms.items():
            if term in query_lower:
                for syn in syns[:2]:  # Limit to 2 synonyms per term
                    expanded_query = query_lower.replace(term, syn)
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)
        
        # Add domain-specific expansions based on detected data types
        if 'financial' in query_info.get('data_types', []):
            expanded.extend([
                query + ' financial data',
                query + ' money revenue',
                query + ' budget cost'
            ])
        
        if 'customer' in query_info.get('data_types', []):
            expanded.extend([
                query + ' customer data',
                query + ' user information',
                query + ' client records'
            ])
        
        # Limit total expansions
        return expanded[:5]
    
    def _hybrid_search(self, query: str, expanded_queries: List[str], query_info: Dict[str, Any], 
                      cursor, max_results: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and keyword matching"""
        
        # Get all chunks with metadata
        cursor.execute("""
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.enhanced_content, c.embedding_id,
                   d.file_path, d.title, d.file_type, d.fileset_name, d.fileset_description,
                   d.schema_info, d.tags, d.user_description, d.file_size
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.indexed_at IS NOT NULL
        """)
        
        chunks = cursor.fetchall()
        results = []
        
        # Generate embeddings for all expanded queries
        query_embeddings = []
        for q in expanded_queries:
            try:
                embedding = self.model.encode([q])[0]
                query_embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to encode expanded query '{q}': {e}")
        
        for chunk_data in chunks:
            (chunk_id, doc_id, chunk_idx, content, enhanced_content, embedding_id, file_path, 
             title, file_type, fileset_name, fileset_description, schema_info, tags, 
             user_description, file_size) = chunk_data
            
            # Load embedding
            embedding_file = self.embeddings_path / f"{embedding_id}.npy"
            if not embedding_file.exists():
                continue
                
            try:
                chunk_embedding = np.load(embedding_file)
                
                # Calculate vector similarities for all query variations
                vector_scores = []
                for query_embedding in query_embeddings:
                    if chunk_embedding.shape == query_embedding.shape:
                        similarity = np.dot(query_embedding, chunk_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                        )
                        vector_scores.append(float(similarity))
                
                if not vector_scores:
                    continue
                
                # Use best vector similarity
                best_vector_score = max(vector_scores)
                
                # Calculate keyword matching score
                keyword_score = self._calculate_keyword_score(query, content, enhanced_content, 
                                                            fileset_name, tags, user_description)
                
                # Calculate metadata relevance score
                metadata_score = self._calculate_metadata_score(query_info, file_type, fileset_name, 
                                                              tags, schema_info)
                
                # Combine scores with weights
                combined_score = (
                    0.6 * best_vector_score +      # Vector similarity (primary)
                    0.25 * keyword_score +         # Keyword matching
                    0.15 * metadata_score          # Metadata relevance
                )
                
                # Apply threshold to combined score
                if combined_score >= similarity_threshold:
                    results.append({
                        'chunk_id': chunk_id,
                        'document_id': doc_id,
                        'file_path': file_path,
                        'title': title,
                        'file_type': file_type,
                        'content': content,
                        'similarity': combined_score,
                        'vector_score': best_vector_score,
                        'keyword_score': keyword_score,
                        'metadata_score': metadata_score,
                        'chunk_index': chunk_idx,
                        'fileset_name': fileset_name or 'Unknown Dataset',
                        'fileset_description': fileset_description or '',
                        'schema_info': schema_info or '',
                        'tags': tags or '',
                        'user_description': user_description or '',
                        'file_size': file_size or 0
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing chunk {chunk_id}: {e}")
                continue
        
        return results
    
    def _semantic_search(self, query: str, expanded_queries: List[str], cursor, 
                        max_results: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Perform pure semantic vector search"""
        # This is similar to the original search but with query expansion
        return self._hybrid_search(query, expanded_queries, {}, cursor, max_results, similarity_threshold)
    
    def _keyword_search(self, query: str, query_info: Dict[str, Any], cursor, max_results: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search using SQL LIKE queries"""
        keywords = query_info.get('keywords', query.split())
        
        # Build SQL query for keyword search
        where_conditions = []
        params = []
        
        for keyword in keywords:
            if len(keyword) > 2:  # Skip very short words
                condition = """(
                    c.content LIKE ? OR c.enhanced_content LIKE ? OR 
                    d.title LIKE ? OR d.fileset_name LIKE ? OR 
                    d.tags LIKE ? OR d.user_description LIKE ?
                )"""
                where_conditions.append(condition)
                keyword_pattern = f"%{keyword}%"
                params.extend([keyword_pattern] * 6)
        
        if not where_conditions:
            return []
        
        sql_query = f"""
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding_id,
                   d.file_path, d.title, d.file_type, d.fileset_name, d.fileset_description,
                   d.schema_info, d.tags, d.user_description
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.indexed_at IS NOT NULL AND ({' OR '.join(where_conditions)})
            LIMIT ?
        """
        
        params.append(max_results * 2)  # Get more results for ranking
        
        cursor.execute(sql_query, params)
        chunks = cursor.fetchall()
        
        results = []
        for chunk_data in chunks:
            (chunk_id, doc_id, chunk_idx, content, embedding_id, file_path, title, file_type,
             fileset_name, fileset_description, schema_info, tags, user_description) = chunk_data
            
            # Calculate keyword relevance score
            keyword_score = self._calculate_keyword_score(query, content, content, 
                                                        fileset_name, tags, user_description)
            
            results.append({
                'chunk_id': chunk_id,
                'document_id': doc_id,
                'file_path': file_path,
                'title': title,
                'file_type': file_type,
                'content': content,
                'similarity': keyword_score,
                'keyword_score': keyword_score,
                'chunk_index': chunk_idx,
                'fileset_name': fileset_name or 'Unknown Dataset',
                'fileset_description': fileset_description or '',
                'schema_info': schema_info or '',
                'tags': tags or '',
                'user_description': user_description or ''
            })
        
        return results
    
    def _calculate_keyword_score(self, query: str, content: str, enhanced_content: str,
                                fileset_name: str, tags: str, user_description: str) -> float:
        """Calculate keyword matching score"""
        query_words = set(query.lower().split())
        
        # Combine all searchable text
        searchable_text = f"{content} {enhanced_content} {fileset_name} {tags} {user_description}".lower()
        
        # Count keyword matches
        matches = 0
        total_words = len(query_words)
        
        for word in query_words:
            if len(word) > 2 and word in searchable_text:
                matches += 1
                
                # Bonus for exact phrase matches
                if word in content.lower():
                    matches += 0.5
                
                # Bonus for matches in metadata
                if word in fileset_name.lower() or word in tags.lower():
                    matches += 0.3
        
        return min(matches / max(total_words, 1), 1.0) if total_words > 0 else 0.0
    
    def _calculate_metadata_score(self, query_info: Dict[str, Any], file_type: str,
                                 fileset_name: str, tags: str, schema_info: str) -> float:
        """Calculate metadata relevance score"""
        score = 0.0
        
        # File type relevance
        data_types = query_info.get('data_types', [])
        if 'technical' in data_types and file_type in ['.py', '.sql', '.json']:
            score += 0.3
        elif 'analytics' in data_types and file_type in ['.csv', '.xlsx']:
            score += 0.3
        
        # Tags relevance
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(',')]
            for data_type in data_types:
                if data_type in tag_list:
                    score += 0.2
        
        # Schema relevance for structured data
        if schema_info and any(dt in ['financial', 'analytics'] for dt in data_types):
            score += 0.2
        
        # Fileset name relevance
        if fileset_name:
            fileset_lower = fileset_name.lower()
            for data_type in data_types:
                if data_type in fileset_lower:
                    score += 0.3
        
        return min(score, 1.0)
    
    def _rerank_results(self, results: List[Dict[str, Any]], query: str, 
                       query_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Re-rank results using multiple signals"""
        
        for result in results:
            # Start with existing similarity score
            base_score = result.get('similarity', 0.0)
            
            # Boost for recent files (if we had timestamps)
            recency_boost = 0.0
            
            # Boost for file size (prefer reasonably sized files)
            size_boost = 0.0
            file_size = result.get('file_size', 0)
            if 1000 < file_size < 1000000:  # 1KB to 1MB sweet spot
                size_boost = 0.05
            
            # Boost for complete metadata
            metadata_boost = 0.0
            if result.get('user_description'):
                metadata_boost += 0.02
            if result.get('tags'):
                metadata_boost += 0.02
            if result.get('schema_info'):
                metadata_boost += 0.02
            
            # Boost for question-answer matching
            qa_boost = 0.0
            if query_info.get('question_type') == 'what' and 'definition' in result.get('content', '').lower():
                qa_boost = 0.1
            elif query_info.get('question_type') == 'how' and any(word in result.get('content', '').lower() 
                                                                 for word in ['step', 'process', 'method']):
                qa_boost = 0.1
            
            # Calculate final score
            result['final_score'] = base_score + recency_boost + size_boost + metadata_boost + qa_boost
        
        # Sort by final score
        results.sort(key=lambda x: x.get('final_score', x.get('similarity', 0)), reverse=True)
        
        return results
    
    def _diversify_results(self, results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Diversify results to avoid too many from the same source"""
        if len(results) <= max_results:
            return results
        
        diversified = []
        source_counts = {}
        
        # First pass: take best result from each unique source
        for result in results:
            source = result.get('fileset_name', 'unknown')
            if source not in source_counts:
                diversified.append(result)
                source_counts[source] = 1
                
                if len(diversified) >= max_results:
                    break
        
        # Second pass: fill remaining slots with next best results
        if len(diversified) < max_results:
            for result in results:
                if result not in diversified:
                    source = result.get('fileset_name', 'unknown')
                    
                    # Limit results per source (max 3)
                    if source_counts.get(source, 0) < 3:
                        diversified.append(result)
                        source_counts[source] = source_counts.get(source, 0) + 1
                        
                        if len(diversified) >= max_results:
                            break
        
        return diversified[:max_results]

    def test_search_functionality(self) -> Dict[str, Any]:
        """Test the search functionality with debug information"""
        try:
            # Test model loading
            self._load_model()
            if self.model is None:
                return {'status': 'failed', 'error': 'Model failed to load'}
            
            # Test database connection
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                return {'status': 'failed', 'error': 'Database not found', 'path': db_path_str}
            
            # Get basic stats
            stats = self.get_index_stats()
            
            # Test a simple search
            test_results = self.search("test", max_results=1, similarity_threshold=0.0)
            
            return {
                'status': 'success',
                'model_loaded': self.model is not None,
                'database_exists': True,
                'stats': stats,
                'test_search_results': len(test_results),
                'test_results_sample': test_results[:1] if test_results else []
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'database_path': str(self.vector_db_path),
                'embeddings_path': str(self.embeddings_path)
            }
