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
        self.max_chunk_size = config_manager.get("vector_search.max_chunk_size", 512)
        self.chunk_overlap = config_manager.get("vector_search.chunk_overlap", 50)
        
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
            logger.info(f"Initializing vector database at: {db_path_str}")
            
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
                self.model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
                
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
        """Extract text content from various file types"""
        file_path = Path(file_path)
        content = ""
        
        try:
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
            elif file_path.suffix.lower() == '.py':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
                    
            elif file_path.suffix.lower() == '.csv':
                # Read first few lines of CSV
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[:100]  # First 100 lines
                    content = ''.join(lines)
                    
            else:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:10000]  # First 10KB
                    
        except Exception as e:
            logger.warning(f"Could not extract text from {file_path}: {e}")
            content = f"File: {file_path.name}\nType: {file_path.suffix}\nContent could not be extracted."
            
        return content
        
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding"""
        if len(text) <= self.max_chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # Try to break at sentence or paragraph boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.max_chunk_size - 100, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
                        
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end - self.chunk_overlap
            if start >= len(text):
                break
                
        return chunks
        
    def _create_enhanced_content(self, chunk: str, title: str, fileset_name: str = None, 
                               fileset_description: str = None, tags: List[str] = None,
                               user_description: str = None, schema_info: str = None) -> str:
        """Create enhanced content for better vector search by including comprehensive metadata"""
        enhanced_parts = []
        
        # Add structured metadata for better search context
        if fileset_name:
            enhanced_parts.append(f"Dataset: {fileset_name}")
        if title:
            enhanced_parts.append(f"Document: {title}")
        if fileset_description:
            enhanced_parts.append(f"Dataset Purpose: {fileset_description}")
        if user_description:
            enhanced_parts.append(f"Description: {user_description}")
        if schema_info:
            enhanced_parts.append(f"Data Structure: {schema_info}")
        if tags:
            enhanced_parts.append(f"Categories: {', '.join(tags)}")
            
        # Add searchable keywords based on metadata
        search_keywords = []
        if fileset_name:
            search_keywords.extend(fileset_name.lower().split('_'))
        if tags:
            search_keywords.extend([tag.lower() for tag in tags])
        if search_keywords:
            enhanced_parts.append(f"Keywords: {' '.join(set(search_keywords))}")
            
        # Add the actual content with clear separation
        enhanced_parts.append("---")  # Separator
        enhanced_parts.append(chunk)
        
        return '\n'.join(enhanced_parts)
        
    def index_document(self, file_path: str, fileset_name: str = None, fileset_description: str = None, 
                      schema_info: str = None, tags: List[str] = None, user_description: str = None) -> bool:
        """Index a single document with enhanced metadata"""
        try:
            file_path = str(Path(file_path).resolve())
            file_hash = self._get_file_hash(file_path)
            
            # Check if document is already indexed and unchanged
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                logger.error(f"Vector database not found at: {db_path_str}")
                return False
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_hash, indexed_at FROM documents WHERE file_path = ?", (file_path,))
            result = cursor.fetchone()
            
            if result and result[0] == file_hash and result[1]:
                logger.debug(f"Document {file_path} already indexed and unchanged")
                conn.close()
                return True
                
            # Extract text content
            content = self._extract_text_content(file_path)
            if not content.strip():
                logger.warning(f"No content extracted from {file_path}")
                conn.close()
                return False
                
            # Get file metadata
            file_stat = Path(file_path).stat()
            file_size = file_stat.st_size
            file_type = Path(file_path).suffix.lower()
            title = Path(file_path).stem
            content_preview = content[:200] + "..." if len(content) > 200 else content
            
            # Detect schema for structured files
            detected_schema = detect_schema(file_path, content)
            final_schema = schema_info or detected_schema
            
            # Auto-generate fileset name if not provided
            if not fileset_name:
                fileset_name = generate_fileset_name(file_path)
                
            # Convert tags to string
            tags_str = ','.join(tags) if tags else ''
            
            # Update or create fileset record
            if fileset_name:
                cursor.execute("""
                    INSERT OR REPLACE INTO filesets 
                    (name, description, schema_info, tags, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (fileset_name, fileset_description, final_schema, tags_str, datetime.now()))
            
            # Insert or update document record with enhanced metadata
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (file_path, file_hash, title, content_preview, file_type, file_size, 
                 fileset_name, fileset_description, schema_info, tags, user_description, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, file_hash, title, content_preview, file_type, file_size,
                  fileset_name, fileset_description, final_schema, tags_str, user_description, datetime.now()))
            
            document_id = cursor.lastrowid
            
            # Update or create fileset record
            if fileset_name:
                cursor.execute("""
                    INSERT OR REPLACE INTO filesets 
                    (name, description, schema_info, tags, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (fileset_name, fileset_description, final_schema, tags_str, datetime.now()))
            
            # Delete existing chunks for this document
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
                    chunk, title, fileset_name, fileset_description, tags, user_description, final_schema
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
                    
            # Update document with indexing info and metadata
            cursor.execute("""
                UPDATE documents 
                SET indexed_at = ?, chunk_count = ?, fileset_name = ?, fileset_description = ?, 
                    schema_info = ?, tags = ?, user_description = ?
                WHERE id = ?
            """, (datetime.now(), len(chunks), fileset_name, fileset_description, 
                  final_schema, tags_str, user_description, document_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Indexed document {file_path} with {len(chunks)} chunks in fileset '{fileset_name}'")
            return True
            
        except Exception as e:
            tbl = traceback.format_exc()
            logger.error(tbl)
            logger.error(f"Error indexing document {file_path}: {e}")
            return False
            
    def index_directory(self, directory_path: str, file_extensions: List[str] = None, 
                       fileset_name: str = None, fileset_description: str = None,
                       tags: List[str] = None) -> Dict[str, Any]:
        """Index all supported files in a directory with dataset metadata"""
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
            'fileset_name': fileset_name
        }
        
        try:
            # First pass: count total files for progress tracking
            all_files = []
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                    all_files.append(file_path)
                    
            results['total_files'] = len(all_files)
            
            # Second pass: index files with metadata
            for file_path in all_files:
                try:
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
        """Update metadata for an existing document"""
        try:
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                return False
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            # Update document metadata
            tags_str = ','.join(tags) if tags else ''
            cursor.execute("""
                UPDATE documents 
                SET fileset_name = ?, fileset_description = ?, tags = ?, user_description = ?, updated_at = ?
                WHERE file_path = ?
            """, (fileset_name, fileset_description, tags_str, user_description, datetime.now(), file_path))
            
            # Update fileset if provided
            if fileset_name:
                cursor.execute("""
                    INSERT OR REPLACE INTO filesets 
                    (name, description, tags, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (fileset_name, fileset_description, tags_str, datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated metadata for document: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            return False
        
    def search(self, query: str, max_results: int = 10, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Perform vector search"""
        if not HAS_VECTOR_DEPS:
            return [{
                'error': 'Vector search dependencies not installed',
                'message': 'Please install: pip install sentence-transformers faiss-cpu numpy'
            }]
            
        try:
            start_time = datetime.now()
            
            # Load model if needed
            self._load_model()
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Get all chunks from database using absolute path
            db_path_str = str(self.vector_db_path.resolve())
            if not Path(db_path_str).exists():
                return [{
                    'message': 'Vector database not found',
                    'suggestion': 'Please rebuild the vector search index'
                }]
                
            conn = sqlite3.connect(db_path_str)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding_id,
                       d.file_path, d.title, d.file_type, d.fileset_name, d.fileset_description,
                       d.schema_info, d.tags, d.user_description
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.indexed_at IS NOT NULL
            """)
            
            chunks = cursor.fetchall()
            
            if not chunks:
                conn.close()
                return [{
                    'message': 'No indexed documents found',
                    'suggestion': 'Please index some documents first'
                }]
                
            # Calculate similarities
            results = []
            
            for chunk_data in chunks:
                (chunk_id, doc_id, chunk_idx, content, embedding_id, file_path, title, file_type,
                 fileset_name, fileset_description, schema_info, tags, user_description) = chunk_data
                
                # Load embedding
                embedding_file = self.embeddings_path / f"{embedding_id}.npy"
                if not embedding_file.exists():
                    continue
                    
                try:
                    chunk_embedding = np.load(embedding_file)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    
                    if similarity >= similarity_threshold:
                        results.append({
                            'chunk_id': chunk_id,
                            'document_id': doc_id,
                            'file_path': file_path,
                            'title': title,
                            'file_type': file_type,
                            'content': content,
                            'similarity': float(similarity),
                            'chunk_index': chunk_idx,
                            'fileset_name': fileset_name or 'Unknown Dataset',
                            'fileset_description': fileset_description or '',
                            'schema_info': schema_info or '',
                            'tags': tags or '',
                            'user_description': user_description or ''
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing embedding {embedding_id}: {e}")
                    continue
                    
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:max_results]
            
            # Log search
            search_time = (datetime.now() - start_time).total_seconds()
            cursor.execute("""
                INSERT INTO search_history (query, search_type, results_count, search_time)
                VALUES (?, ?, ?, ?)
            """, (query, 'vector_search', len(results), search_time))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Vector search for '{query}' returned {len(results)} results in {search_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return [{
                'error': 'Search failed',
                'message': str(e)
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
            
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
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
                        
            logger.info("Vector search index cleared")
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise
            
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
                    'status': 'Database not found'
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
            
            conn.close()
            
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
            
            return {
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'total_size_bytes': total_size,
                'file_types': file_types,
                'recent_indexing_count': recent_count,
                'embeddings_path': str(self.embeddings_path),
                'database_path': str(self.vector_db_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
