"""
Settings Dialog
Configuration interface for AI tools, databases, and application preferences
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, 
    QFormLayout, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QLabel, QGroupBox, QFileDialog, QMessageBox, QTextEdit,
    QSpinBox, QSlider, QScrollArea, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AIToolsTab(QWidget):
    """Tab for configuring AI tools"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Setup AI tools configuration UI"""
        layout = QVBoxLayout(self)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Anthropic Claude
        claude_group = QGroupBox("Anthropic Claude")
        claude_layout = QFormLayout(claude_group)
        
        self.claude_enabled = QCheckBox("Enable Claude")
        claude_layout.addRow(self.claude_enabled)
        
        self.claude_api_key = QLineEdit()
        self.claude_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.claude_api_key.setPlaceholderText("sk-ant-...")
        claude_layout.addRow("API Key:", self.claude_api_key)
        
        self.claude_model = QComboBox()
        self.claude_model.addItems([
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229", 
            "claude-3-haiku-20240307"
        ])
        claude_layout.addRow("Model:", self.claude_model)
        
        scroll_layout.addWidget(claude_group)
        
        # OpenAI GPT
        openai_group = QGroupBox("OpenAI GPT")
        openai_layout = QFormLayout(openai_group)
        
        self.openai_enabled = QCheckBox("Enable GPT")
        openai_layout.addRow(self.openai_enabled)
        
        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_api_key.setPlaceholderText("sk-...")
        openai_layout.addRow("API Key:", self.openai_api_key)
        
        self.openai_model = QComboBox()
        self.openai_model.addItems([
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
        ])
        openai_layout.addRow("Model:", self.openai_model)
        
        scroll_layout.addWidget(openai_group)
        
        # Local Models
        local_group = QGroupBox("Local Models")
        local_layout = QFormLayout(local_group)
        
        self.local_enabled = QCheckBox("Enable Local Models")
        local_layout.addRow(self.local_enabled)
        
        self.local_model = QComboBox()
        self.local_model.addItems([
            # Large models (7B+)
            "qwen2.5:14b", "qwen2.5:7b", "qwen2.5:3b", 
            "llama3.2:3b", "llama3.1:8b", "llama3.1:7b",
            "gemma2:9b", "gemma2:7b", "gemma2:2b",
            "mistral:7b", "mistral-nemo:12b",
            "phi3.5:3.8b", "phi3:3.8b", "phi3:mini",
            # Gemma 3 models (<5B)
            "gemma3:2b", "gemma3:1.5b", "gemma3:1b",
            # Medium models (1-3B)
            "qwen2.5:1.5b", "qwen2.5:0.5b",
            "llama3.2:1b", "gemma2:2b",
            "phi3:mini", "phi3.5:mini",
            "tinyllama:1.1b", "stablelm2:1.6b",
            # Small models (<1B)
            "qwen2.5:0.5b", "smollm:360m", "smollm:135m",
            "tinydolphin:1.1b", "all-minilm:22m",
            "nomic-embed-text:137m"
        ])
        local_layout.addRow("Default Model:", self.local_model)
        
        # Model verification button
        verify_model_btn = QPushButton("Verify Model")
        verify_model_btn.setToolTip("Check if the selected model is available in Ollama")
        verify_model_btn.clicked.connect(self.verify_local_model)
        verify_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #138496; }
        """)
        local_layout.addRow("", verify_model_btn)
        
        self.local_endpoint = QLineEdit()
        self.local_endpoint.setPlaceholderText("http://localhost:11434")
        local_layout.addRow("Ollama Endpoint:", self.local_endpoint)
        
        # Advanced local model settings
        advanced_local = QGroupBox("Advanced Local Model Settings")
        advanced_layout = QFormLayout(advanced_local)
        
        self.context_length = QSpinBox()
        self.context_length.setRange(512, 32768)
        self.context_length.setValue(4096)
        self.context_length.setSuffix(" tokens")
        advanced_layout.addRow("Context Length:", self.context_length)
        
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setValue(0.7)
        self.temperature.setSingleStep(0.1)
        self.temperature.setDecimals(1)
        advanced_layout.addRow("Temperature:", self.temperature)
        
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(50, 4096)
        self.max_tokens.setValue(512)
        self.max_tokens.setSuffix(" tokens")
        advanced_layout.addRow("Max Response Tokens:", self.max_tokens)
        
        # GPU acceleration option
        self.use_gpu = QCheckBox("Use GPU acceleration (if available)")
        self.use_gpu.setChecked(True)
        advanced_layout.addRow(self.use_gpu)
        
        scroll_layout.addWidget(advanced_local)
        
        scroll_layout.addWidget(local_group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
    def load_settings(self):
        """Load AI tool settings from config"""
        # Claude settings
        claude_config = self.config_manager.get_ai_tool_config("anthropic")
        self.claude_enabled.setChecked(claude_config.get("enabled", False))
        self.claude_api_key.setText(claude_config.get("api_key", ""))
        self.claude_model.setCurrentText(claude_config.get("model", "claude-3-sonnet-20240229"))
        
        # OpenAI settings
        openai_config = self.config_manager.get_ai_tool_config("openai")
        self.openai_enabled.setChecked(openai_config.get("enabled", False))
        self.openai_api_key.setText(openai_config.get("api_key", ""))
        self.openai_model.setCurrentText(openai_config.get("model", "gpt-4"))
        
        # Local model settings
        local_config = self.config_manager.get_ai_tool_config("local_models")
        self.local_enabled.setChecked(local_config.get("enabled", False))
        self.local_model.setCurrentText(local_config.get("default_model", "qwen2.5:3b"))
        self.local_endpoint.setText(local_config.get("endpoint", "http://localhost:11434"))
        
        # Advanced local settings
        self.context_length.setValue(local_config.get("context_length", 4096))
        self.temperature.setValue(local_config.get("temperature", 0.7))
        self.max_tokens.setValue(local_config.get("max_tokens", 512))
        self.use_gpu.setChecked(local_config.get("use_gpu", True))
        
    def verify_local_model(self):
        """Verify that the selected local model is available"""
        model_name = self.local_model.currentText()
        endpoint = self.local_endpoint.text() or "http://localhost:11434"
        
        try:
            import requests
            import json
            
            self.model_status_label.setText("🔄 Checking model availability...")
            self.model_status_label.setStyleSheet("color: #ffc107;")
            
            # Check if Ollama is running
            try:
                response = requests.get(f"{endpoint}/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    
                    if model_name in available_models:
                        self.model_status_label.setText(f"✅ {model_name} is available and ready")
                        self.model_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
                    else:
                        # Show available models for reference
                        available_list = ", ".join(available_models[:5])
                        if len(available_models) > 5:
                            available_list += f" (+{len(available_models) - 5} more)"
                        
                        self.model_status_label.setText(
                            f"⚠️ {model_name} not found. Available: {available_list}\n"
                            f"Run: ollama pull {model_name}"
                        )
                        self.model_status_label.setStyleSheet("color: #ffc107;")
                else:
                    self.model_status_label.setText(f"❌ Ollama API error: {response.status_code}")
                    self.model_status_label.setStyleSheet("color: #dc3545;")
                    
            except requests.exceptions.ConnectionError:
                self.model_status_label.setText(
                    f"❌ Cannot connect to Ollama at {endpoint}\n"
                    "Make sure Ollama is running: ollama serve"
                )
                self.model_status_label.setStyleSheet("color: #dc3545;")
            except requests.exceptions.Timeout:
                self.model_status_label.setText("⏱️ Connection timeout - Ollama may be starting up")
                self.model_status_label.setStyleSheet("color: #ffc107;")
                
        except ImportError:
            self.model_status_label.setText("❌ 'requests' library required for verification\nRun: pip install requests")
            self.model_status_label.setStyleSheet("color: #dc3545;")
        except Exception as e:
            self.model_status_label.setText(f"❌ Verification error: {str(e)}")
            self.model_status_label.setStyleSheet("color: #dc3545;")
    
    def save_settings(self):
        """Save AI tool settings to config"""
        # Claude settings
        self.config_manager.set("ai_tools.anthropic.enabled", self.claude_enabled.isChecked())
        self.config_manager.set("ai_tools.anthropic.api_key", self.claude_api_key.text())
        self.config_manager.set("ai_tools.anthropic.model", self.claude_model.currentText())
        
        # OpenAI settings
        self.config_manager.set("ai_tools.openai.enabled", self.openai_enabled.isChecked())
        self.config_manager.set("ai_tools.openai.api_key", self.openai_api_key.text())
        self.config_manager.set("ai_tools.openai.model", self.openai_model.currentText())
        
        # Local model settings
        self.config_manager.set("ai_tools.local_models.enabled", self.local_enabled.isChecked())
        self.config_manager.set("ai_tools.local_models.default_model", self.local_model.currentText())
        self.config_manager.set("ai_tools.local_models.endpoint", self.local_endpoint.text())
        self.config_manager.set("ai_tools.local_models.context_length", self.context_length.value())
        self.config_manager.set("ai_tools.local_models.temperature", self.temperature.value())
        self.config_manager.set("ai_tools.local_models.max_tokens", self.max_tokens.value())
        self.config_manager.set("ai_tools.local_models.use_gpu", self.use_gpu.isChecked())

class DatabaseTab(QWidget):
    """Tab for configuring database connections"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Setup database configuration UI"""
        layout = QVBoxLayout(self)
        
        # Default backend
        backend_group = QGroupBox("Default Local Backend")
        backend_layout = QFormLayout(backend_group)
        
        self.default_backend = QComboBox()
        self.default_backend.addItems(["sqlite", "duckdb"])
        backend_layout.addRow("Backend:", self.default_backend)
        
        layout.addWidget(backend_group)
        
        # Vector Database
        vector_group = QGroupBox("Vector Search Database")
        vector_layout = QFormLayout(vector_group)
        
        self.vector_db_type = QComboBox()
        self.vector_db_type.addItems(["internal", "chroma", "pinecone", "weaviate"])
        vector_layout.addRow("Type:", self.vector_db_type)
        
        self.vector_db_path = QLineEdit()
        self.vector_db_path.setPlaceholderText("/path/to/vector/db")
        vector_layout.addRow("Path/URL:", self.vector_db_path)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_vector_db)
        vector_layout.addRow("", browse_btn)
        
        layout.addWidget(vector_group)
        
        # External Connections
        connections_group = QGroupBox("External Database Connections")
        connections_layout = QVBoxLayout(connections_group)
        
        # Add connection button
        add_conn_btn = QPushButton("Add New Connection")
        add_conn_btn.clicked.connect(self.add_database_connection)
        connections_layout.addWidget(add_conn_btn)
        
        # Connection list (simplified for skeleton)
        self.connections_label = QLabel("No external connections configured")
        self.connections_label.setStyleSheet("color: #666; font-style: italic;")
        connections_layout.addWidget(self.connections_label)
        
        layout.addWidget(connections_group)
        
        layout.addStretch()
        
    def load_settings(self):
        """Load database settings from config"""
        self.default_backend.setCurrentText(
            self.config_manager.get("databases.default_backend", "sqlite")
        )
        
        self.vector_db_type.setCurrentText(
            self.config_manager.get("vector_search.database_type", "internal")
        )
        
        self.vector_db_path.setText(
            self.config_manager.get("vector_search.database_path", "")
        )
        
        # Update connections display
        connections = self.config_manager.get_database_connections()
        if connections:
            self.connections_label.setText(f"{len(connections)} connections configured")
        
    def save_settings(self):
        """Save database settings to config"""
        self.config_manager.set("databases.default_backend", self.default_backend.currentText())
        self.config_manager.set("vector_search.database_type", self.vector_db_type.currentText())
        self.config_manager.set("vector_search.database_path", self.vector_db_path.text())
        
    def browse_vector_db(self):
        """Browse for vector database path"""
        if self.vector_db_type.currentText() == "internal":
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Select Vector Database Location", "", "Database Files (*.db);;All Files (*)"
            )
            if file_path:
                self.vector_db_path.setText(file_path)
        
    def add_database_connection(self):
        """Add new database connection"""
        dialog = DatabaseConnectionDialog(self)
        if dialog.exec():
            connection_config = dialog.get_connection_config()
            self.config_manager.add_database_connection(connection_config)
            self.load_settings()  # Refresh display

class VectorSearchTab(QWidget):
    """Tab for configuring vector search settings"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Setup vector search configuration UI"""
        layout = QVBoxLayout(self)
        
        # Database Configuration
        db_group = QGroupBox("Database Configuration")
        db_layout = QFormLayout(db_group)
        
        # Database path
        db_path_layout = QHBoxLayout()
        self.database_path = QLineEdit()
        self.database_path.setPlaceholderText("Path to vector search database")
        self.database_path.textChanged.connect(self.update_status)
        db_path_layout.addWidget(self.database_path)
        
        browse_db_btn = QPushButton("Browse...")
        browse_db_btn.clicked.connect(self.browse_database_path)
        db_path_layout.addWidget(browse_db_btn)
        
        create_db_btn = QPushButton("Create New")
        create_db_btn.setToolTip("Create a new vector database file")
        create_db_btn.clicked.connect(self.create_new_database)
        create_db_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        db_path_layout.addWidget(create_db_btn)
        
        reset_db_btn = QPushButton("Reset")
        reset_db_btn.setToolTip("Reset to default path")
        reset_db_btn.clicked.connect(self.reset_database_path)
        db_path_layout.addWidget(reset_db_btn)
        
        db_layout.addRow("Database Path:", db_path_layout)
        
        # Embeddings path
        embed_path_layout = QHBoxLayout()
        self.embeddings_path = QLineEdit()
        self.embeddings_path.setPlaceholderText("Path to embeddings storage")
        self.embeddings_path.textChanged.connect(self.update_status)
        embed_path_layout.addWidget(self.embeddings_path)
        
        browse_embed_btn = QPushButton("Browse...")
        browse_embed_btn.clicked.connect(self.browse_embeddings_path)
        embed_path_layout.addWidget(browse_embed_btn)
        
        create_embed_btn = QPushButton("Create Dir")
        create_embed_btn.setToolTip("Create embeddings directory")
        create_embed_btn.clicked.connect(self.create_embeddings_directory)
        create_embed_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #138496; }
        """)
        embed_path_layout.addWidget(create_embed_btn)
        
        reset_embed_btn = QPushButton("Reset")
        reset_embed_btn.setToolTip("Reset to default path")
        reset_embed_btn.clicked.connect(self.reset_embeddings_path)
        embed_path_layout.addWidget(reset_embed_btn)
        
        db_layout.addRow("Embeddings Path:", embed_path_layout)
        
        layout.addWidget(db_group)
        
        # Embedding Model
        embedding_group = QGroupBox("Embedding Model")
        embedding_layout = QFormLayout(embedding_group)
        
        self.embedding_model = QComboBox()
        self.embedding_model.addItems([
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "text-embedding-ada-002",
            "embed-english-v3.0"
        ])
        embedding_layout.addRow("Model:", self.embedding_model)
        
        layout.addWidget(embedding_group)
        
        # Chunking Settings
        chunking_group = QGroupBox("Text Chunking")
        chunking_layout = QFormLayout(chunking_group)
        
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(100, 5000)
        self.chunk_size.setValue(1000)
        chunking_layout.addRow("Chunk Size:", self.chunk_size)
        
        self.chunk_overlap = QSpinBox()
        self.chunk_overlap.setRange(0, 1000)
        self.chunk_overlap.setValue(200)
        chunking_layout.addRow("Overlap:", self.chunk_overlap)
        
        layout.addWidget(chunking_group)
        
        # Search Settings
        search_group = QGroupBox("Search Settings")
        search_layout = QFormLayout(search_group)
        
        self.max_results = QSpinBox()
        self.max_results.setRange(1, 100)
        self.max_results.setValue(10)
        search_layout.addRow("Max Results:", self.max_results)
        
        self.similarity_threshold = QSlider(Qt.Orientation.Horizontal)
        self.similarity_threshold.setRange(0, 100)
        self.similarity_threshold.setValue(70)
        search_layout.addRow("Similarity Threshold:", self.similarity_threshold)
        
        layout.addWidget(search_group)
        
        # Status and Testing
        status_group = QGroupBox("Status & Actions")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Vector search status will be shown here")
        self.status_label.setStyleSheet("color: #666; font-style: italic; padding: 8px;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        # Action buttons layout
        actions_layout = QHBoxLayout()
        
        test_btn = QPushButton("Test Configuration")
        test_btn.clicked.connect(self.test_configuration)
        test_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0056b3; }
        """)
        actions_layout.addWidget(test_btn)
        
        initialize_btn = QPushButton("Initialize Default Database")
        initialize_btn.clicked.connect(self.initialize_default_database)
        initialize_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        actions_layout.addWidget(initialize_btn)
        
        status_layout.addLayout(actions_layout)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
    def browse_database_path(self):
        """Browse for database file location"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Vector Database Location",
            self.database_path.text() or str(Path.cwd() / "data" / "vector_search.db"),
            "Database Files (*.db);;All Files (*)"
        )
        if file_path:
            self.database_path.setText(file_path)
            self.update_status()
            
    def browse_embeddings_path(self):
        """Browse for embeddings directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Embeddings Directory",
            self.embeddings_path.text() or str(Path.cwd() / "data" / "embeddings")
        )
        if dir_path:
            self.embeddings_path.setText(dir_path)
            self.update_status()
            
    def create_new_database(self):
        """Create a new vector database file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Create New Vector Database",
            str(Path.cwd() / "data" / "vector_search.db"),
            "Database Files (*.db);;All Files (*)"
        )
        
        if file_path:
            try:
                # Ensure the directory exists
                db_path = Path(file_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create the database file by initializing it
                from utils.vector_search import VectorSearchEngine
                
                # Temporarily set the path in config
                old_path = self.config_manager.get("vector_search.database_path")
                self.config_manager.set("vector_search.database_path", str(db_path))
                
                # Initialize the database
                vector_engine = VectorSearchEngine(self.config_manager)
                
                # Restore old path temporarily
                self.config_manager.set("vector_search.database_path", old_path)
                
                # Set the new path in the UI
                self.database_path.setText(str(db_path))
                self.update_status()
                
                QMessageBox.information(
                    self,
                    "Database Created",
                    f"Vector database successfully created at:\n{db_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Creating Database",
                    f"Failed to create vector database:\n{str(e)}"
                )
                
    def create_embeddings_directory(self):
        """Create embeddings directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Location for Embeddings Directory",
            str(Path.cwd() / "data")
        )
        
        if dir_path:
            try:
                embeddings_path = Path(dir_path) / "embeddings"
                embeddings_path.mkdir(parents=True, exist_ok=True)
                
                self.embeddings_path.setText(str(embeddings_path))
                self.update_status()
                
                QMessageBox.information(
                    self,
                    "Directory Created",
                    f"Embeddings directory created at:\n{embeddings_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Creating Directory",
                    f"Failed to create embeddings directory:\n{str(e)}"
                )
        
    def reset_database_path(self):
        """Reset database path to default"""
        default_path = str(Path.cwd() / "data" / "vector_search.db")
        self.database_path.setText(default_path)
        self.update_status()
        
    def reset_embeddings_path(self):
        """Reset embeddings path to default"""
        default_path = str(Path.cwd() / "data" / "embeddings")
        self.embeddings_path.setText(default_path)
        self.update_status()
        
    def test_configuration(self):
        """Test the vector search configuration"""
        try:
            from utils.vector_search import VectorSearchEngine
            
            # Temporarily update config
            old_db_path = self.config_manager.get("vector_search.database_path")
            old_embed_path = self.config_manager.get("vector_search.embeddings_path")
            
            self.config_manager.set("vector_search.database_path", self.database_path.text())
            self.config_manager.set("vector_search.embeddings_path", self.embeddings_path.text())
            
            # Test creating vector engine
            vector_engine = VectorSearchEngine(self.config_manager)
            stats = vector_engine.get_index_stats()
            
            self.status_label.setText(f"✅ Configuration valid. Database accessible.")
            self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
            # Restore old config
            self.config_manager.set("vector_search.database_path", old_db_path)
            self.config_manager.set("vector_search.embeddings_path", old_embed_path)
            
        except Exception as e:
            self.status_label.setText(f"❌ Configuration error: {str(e)}")
            self.status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            
    def initialize_default_database(self):
        """Initialize default vector database and embeddings directory"""
        try:
            # Set default paths
            default_data_dir = Path.cwd() / "data"
            default_db_path = default_data_dir / "vector_search.db"
            default_embed_path = default_data_dir / "embeddings"
            
            # Create directories
            default_data_dir.mkdir(parents=True, exist_ok=True)
            default_embed_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            from utils.vector_search import VectorSearchEngine
            
            # Temporarily set paths in config
            old_db_path = self.config_manager.get("vector_search.database_path")
            old_embed_path = self.config_manager.get("vector_search.embeddings_path")
            
            self.config_manager.set("vector_search.database_path", str(default_db_path))
            self.config_manager.set("vector_search.embeddings_path", str(default_embed_path))
            
            # Create the vector engine (this will initialize the database)
            vector_engine = VectorSearchEngine(self.config_manager)
            
            # Restore old paths temporarily
            self.config_manager.set("vector_search.database_path", old_db_path)
            self.config_manager.set("vector_search.embeddings_path", old_embed_path)
            
            # Update UI with new paths
            self.database_path.setText(str(default_db_path))
            self.embeddings_path.setText(str(default_embed_path))
            self.update_status()
            
            QMessageBox.information(
                self,
                "Default Database Initialized",
                f"Default vector search database and embeddings directory created:\n\n"
                f"Database: {default_db_path}\n"
                f"Embeddings: {default_embed_path}\n\n"
                f"You can now start indexing documents!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize default database:\n{str(e)}"
            )
    
    def update_status(self):
        """Update the status display"""
        try:
            from pathlib import Path
            db_path_text = self.database_path.text().strip()
            embed_path_text = self.embeddings_path.text().strip()
            
            if not db_path_text and not embed_path_text:
                self.status_label.setText("⚠️ No paths configured. Click 'Initialize Default Database' to get started.")
                self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
                return
                
            status_parts = []
            
            if db_path_text:
                db_path = Path(db_path_text)
                if db_path.exists():
                    status_parts.append(f"✅ Database exists: {db_path.name}")
                else:
                    status_parts.append(f"📁 Database location: {db_path.name} (will be created)")
                    
            if embed_path_text:
                embed_path = Path(embed_path_text)
                if embed_path.exists():
                    status_parts.append(f"✅ Embeddings directory exists: {embed_path.name}")
                else:
                    status_parts.append(f"📁 Embeddings location: {embed_path.name} (will be created)")
            
            if status_parts:
                self.status_label.setText(" | ".join(status_parts))
                # Check if both paths exist
                db_exists = Path(db_path_text).exists() if db_path_text else False
                embed_exists = Path(embed_path_text).exists() if embed_path_text else False
                
                if db_exists and embed_exists:
                    self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
                elif db_exists or embed_exists:
                    self.status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
                else:
                    self.status_label.setStyleSheet("color: #17a2b8; font-weight: bold;")
            else:
                self.status_label.setText("⚠️ Please configure database and embeddings paths")
                self.status_label.setStyleSheet("color: #ffc107;")
                
        except Exception as e:
            self.status_label.setText(f"❌ Error checking paths: {str(e)}")
            self.status_label.setStyleSheet("color: #dc3545;")
        
    def load_settings(self):
        """Load vector search settings from config"""
        from pathlib import Path
        
        default_db_path = str(Path.cwd() / "data" / "vector_search.db")
        default_embed_path = str(Path.cwd() / "data" / "embeddings")
        
        # Load paths from config, but don't set defaults if they're empty
        db_path = self.config_manager.get("vector_search.database_path", "")
        embed_path = self.config_manager.get("vector_search.embeddings_path", "")
        
        # Only set default paths if nothing is configured
        if not db_path:
            db_path = default_db_path
        if not embed_path:
            embed_path = default_embed_path
            
        self.database_path.setText(db_path)
        self.embeddings_path.setText(embed_path)
        
        self.embedding_model.setCurrentText(
            self.config_manager.get("vector_search.model", "all-MiniLM-L6-v2")
        )
        
        self.chunk_size.setValue(
            self.config_manager.get("vector_search.max_chunk_size", 512)
        )
        
        self.chunk_overlap.setValue(
            self.config_manager.get("vector_search.chunk_overlap", 50)
        )
        
        # Update status
        self.update_status()
        
    def save_settings(self):
        """Save vector search settings to config"""
        self.config_manager.set("vector_search.database_path", self.database_path.text())
        self.config_manager.set("vector_search.embeddings_path", self.embeddings_path.text())
        self.config_manager.set("vector_search.model", self.embedding_model.currentText())
        self.config_manager.set("vector_search.max_chunk_size", self.chunk_size.value())
        self.config_manager.set("vector_search.chunk_overlap", self.chunk_overlap.value())

class DatabaseConnectionDialog(QDialog):
    """Dialog for adding/editing database connections"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup database connection dialog"""
        self.setWindowTitle("Add Database Connection")
        self.setModal(True)
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        self.connection_name = QLineEdit()
        self.connection_name.setPlaceholderText("My Database")
        form_layout.addRow("Name:", self.connection_name)
        
        self.db_type = QComboBox()
        self.db_type.addItems(["PostgreSQL", "MySQL", "SQL Server", "BigQuery", "Snowflake"])
        form_layout.addRow("Type:", self.db_type)
        
        self.host = QLineEdit()
        self.host.setPlaceholderText("localhost")
        form_layout.addRow("Host:", self.host)
        
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(5432)
        form_layout.addRow("Port:", self.port)
        
        self.database = QLineEdit()
        form_layout.addRow("Database:", self.database)
        
        self.username = QLineEdit()
        form_layout.addRow("Username:", self.username)
        
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addRow("Password:", self.password)
        
        self.ssl_verify = QCheckBox("Verify SSL Certificate")
        self.ssl_verify.setChecked(True)
        form_layout.addRow(self.ssl_verify)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self.test_connection)
        button_layout.addWidget(test_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
    def test_connection(self):
        """Test database connection"""
        # Placeholder for connection testing
        QMessageBox.information(self, "Test Result", "Connection test would be performed here.")
        
    def get_connection_config(self):
        """Get connection configuration"""
        return {
            'name': self.connection_name.text(),
            'type': self.db_type.currentText(),
            'host': self.host.text(),
            'port': self.port.value(),
            'database': self.database.text(),
            'username': self.username.text(),
            'password': self.password.text(),
            'ssl_verify': self.ssl_verify.isChecked()
        }

class SettingsDialog(QDialog):
    """Main settings dialog with tabs for different configuration areas"""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the settings dialog UI"""
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # AI Tools tab
        self.ai_tab = AIToolsTab(self.config_manager)
        self.tab_widget.addTab(self.ai_tab, "AI Tools")
        
        # Database tab
        self.db_tab = DatabaseTab(self.config_manager)
        self.tab_widget.addTab(self.db_tab, "Databases")
        
        # Vector Search tab
        self.vector_tab = VectorSearchTab(self.config_manager)
        self.tab_widget.addTab(self.vector_tab, "Vector Search")
        
        layout.addWidget(self.tab_widget)
        
        # Add cloud authentication shortcut
        cloud_auth_button = QPushButton("Configure Cloud Authentication")
        cloud_auth_button.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        cloud_auth_button.clicked.connect(self.show_cloud_auth)
        layout.addWidget(cloud_auth_button)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        defaults_btn = QPushButton("Reset to Defaults")
        defaults_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(defaults_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(apply_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
    def apply_settings(self):
        """Apply settings without closing dialog"""
        self.ai_tab.save_settings()
        self.db_tab.save_settings()
        self.vector_tab.save_settings()
        self.settings_changed.emit()
        
    def show_cloud_auth(self):
        """Show cloud authentication dialog"""
        from ui.dialogs.cloud_auth_dialog import CloudAuthDialog
        
        dialog = CloudAuthDialog(self)
        dialog.exec()
    
    def accept(self):
        """Accept and save settings"""
        self.apply_settings()
        super().accept()
        
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self, 
            "Reset Settings", 
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset config to defaults
            self.config_manager.config_data = self.config_manager.get_default_config()
            
            # Reload UI
            self.ai_tab.load_settings()
            self.db_tab.load_settings()
            self.vector_tab.load_settings()
            
            self.settings_changed.emit()
