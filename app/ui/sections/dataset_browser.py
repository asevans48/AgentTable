"""
Dataset Browser Widget
Shows registered datasets with filtering and access information
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QLineEdit, QFrame, QScrollArea,
    QMessageBox, QDialog, QTextEdit, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import List, Dict, Any
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetItem(QFrame):
    """Individual dataset item widget"""
    
    dataset_selected = pyqtSignal(dict)  # dataset_info
    chat_requested = pyqtSignal(dict)  # dataset_info
    access_requested = pyqtSignal(dict)  # dataset_info
    
    def __init__(self, dataset_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.dataset_info = dataset_info
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dataset item UI"""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin: 1px;
                padding: 8px;
            }
            QFrame:hover {
                border-color: #1a73e8;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Header with name and type
        header_layout = QHBoxLayout()
        
        # Dataset name
        name = self.dataset_info.get('name', 'Unnamed Dataset')
        self.name_label = QLabel(f"<b>{name}</b>")
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("color: #1a73e8; font-size: 11pt;")
        header_layout.addWidget(self.name_label, 1)
        
        # Dataset type badge
        dataset_type = self.dataset_info.get('type', 'Unknown')
        type_badge = QLabel(dataset_type)
        type_badge.setStyleSheet("""
            QLabel {
                background-color: #e8f0fe;
                color: #1a73e8;
                padding: 2px 6px;
                border-radius: 8px;
                font-size: 8pt;
                font-weight: bold;
            }
        """)
        type_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(type_badge)
        
        layout.addLayout(header_layout)
        
        # Compact metadata row
        metadata_layout = QHBoxLayout()
        
        # Owner
        owner = self.dataset_info.get('owner', 'Unknown')
        owner_label = QLabel(f"👤 {owner}")
        owner_label.setStyleSheet("color: #666; font-size: 8pt;")
        metadata_layout.addWidget(owner_label)
        
        metadata_layout.addStretch()
        
        # Row count (if available) - more compact
        row_count = self.dataset_info.get('row_count')
        if row_count is not None:
            if row_count >= 1000000:
                count_str = f"{row_count // 1000000}M"
            elif row_count >= 1000:
                count_str = f"{row_count // 1000}K"
            else:
                count_str = str(row_count)
            count_label = QLabel(f"📊 {count_str}")
            count_label.setStyleSheet("color: #666; font-size: 8pt;")
            metadata_layout.addWidget(count_label)
            
        layout.addLayout(metadata_layout)
        
        # Access and action buttons
        actions_layout = QHBoxLayout()
        
        # Access level indicator
        access_level = self.dataset_info.get('access_level', 'Unknown')
        has_access = access_level in ['Full', 'Read', 'Read-Only']
        
        if has_access:
            access_color = '#4caf50' if access_level == 'Full' else '#ff9800'
            access_text = f"✓ {access_level}"
        else:
            access_color = '#f44336'
            access_text = "✗ No Access"
            
        access_label = QLabel(access_text)
        access_label.setStyleSheet(f"""
            QLabel {{
                color: {access_color};
                font-size: 8pt;
                font-weight: bold;
                padding: 2px 4px;
            }}
        """)
        actions_layout.addWidget(access_label)
        
        actions_layout.addStretch()
        
        # Action buttons
        if has_access and self.dataset_info.get('supports_chat', True):
            chat_btn = QPushButton("Chat")
            chat_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1a73e8;
                    color: white;
                    border: none;
                    padding: 3px 8px;
                    border-radius: 3px;
                    font-size: 8pt;
                }
                QPushButton:hover { background-color: #1557b0; }
            """)
            chat_btn.clicked.connect(self.on_chat_clicked)
            actions_layout.addWidget(chat_btn)
        
        view_btn = QPushButton("View" if has_access else "Request")
        btn_color = "#34a853" if has_access else "#ff9800"
        btn_hover = "#2d8f47" if has_access else "#e68900"
        
        view_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {btn_color};
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 8pt;
            }}
            QPushButton:hover {{ background-color: {btn_hover}; }}
        """)
        
        if has_access:
            view_btn.clicked.connect(self.on_view_clicked)
        else:
            view_btn.clicked.connect(self.on_request_access_clicked)
            
        actions_layout.addWidget(view_btn)
        
        layout.addLayout(actions_layout)
        
        # Make the whole item clickable
        self.mousePressEvent = self.on_item_clicked
        
    def on_item_clicked(self, event):
        """Handle item click"""
        self.dataset_selected.emit(self.dataset_info)
        
    def on_chat_clicked(self):
        """Handle chat button click"""
        self.chat_requested.emit(self.dataset_info)
        
    def on_view_clicked(self):
        """Handle view button click"""
        self.dataset_selected.emit(self.dataset_info)
        
    def on_request_access_clicked(self):
        """Handle request access button click"""
        self.access_requested.emit(self.dataset_info)

class DatasetBrowser(QWidget):
    """Dataset browser widget"""
    
    dataset_selected = pyqtSignal(dict)  # dataset_info
    chat_requested = pyqtSignal(dict)  # dataset_info
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.all_datasets = []
        self.filtered_datasets = []
        self.current_filters = {}
        self.vector_engine = None
        self.connection_manager = None
        
        self.setup_ui()
        self.setup_connections()
        self.load_vector_engine()
        self.load_connection_manager()
        self.load_datasets()
    
    def load_connection_manager(self):
        """Load database connection manager"""
        try:
            from utils.database.connection_manager import DatabaseConnectionManager
            from utils.credential_manager import CredentialManager
            
            credential_manager = CredentialManager()
            self.connection_manager = DatabaseConnectionManager(self.config_manager, credential_manager)
            logger.info("Database connection manager loaded")
        except Exception as e:
            logger.error(f"Failed to load connection manager: {e}")
            self.connection_manager = None
        
    def setup_ui(self):
        """Setup the dataset browser UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Datasets")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("Refresh datasets")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                background: white;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Search and filters
        search_layout = QVBoxLayout()
        
        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search datasets...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 9pt;
            }
        """)
        search_layout.addWidget(self.search_input)
        
        # Filter row
        filter_layout = QHBoxLayout()
        
        # Type filter
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All Types", "Table", "View", "File", "API", "Stream"])
        self.type_filter.setStyleSheet("font-size: 8pt;")
        filter_layout.addWidget(self.type_filter)
        
        # Access filter
        self.access_filter = QComboBox()
        self.access_filter.addItems(["All Access", "Full Access", "Read Access", "No Access"])
        self.access_filter.setStyleSheet("font-size: 8pt;")
        filter_layout.addWidget(self.access_filter)
        
        search_layout.addLayout(filter_layout)
        layout.addLayout(search_layout)
        
        # Dataset list
        self.dataset_scroll = QScrollArea()
        self.dataset_scroll.setWidgetResizable(True)
        self.dataset_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.dataset_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dataset_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
        """)
        
        self.datasets_container = QWidget()
        self.datasets_layout = QVBoxLayout(self.datasets_container)
        self.datasets_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.datasets_layout.setSpacing(2)
        
        self.dataset_scroll.setWidget(self.datasets_container)
        layout.addWidget(self.dataset_scroll)
        
        # Status label
        self.status_label = QLabel("Loading datasets...")
        self.status_label.setStyleSheet("font-size: 8pt; color: #666; padding: 4px;")
        layout.addWidget(self.status_label)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.search_input.textChanged.connect(self.apply_filters)
        self.type_filter.currentTextChanged.connect(self.apply_filters)
        self.access_filter.currentTextChanged.connect(self.apply_filters)
        
    def load_vector_engine(self):
        """Load vector search engine for automatic indexing"""
        try:
            from utils.vector_search import VectorSearchEngine
            self.vector_engine = VectorSearchEngine(self.config_manager)
            logger.info("Vector search engine loaded for dataset browser")
        except ImportError:
            logger.warning("Vector search dependencies not available")
            self.vector_engine = None
        except Exception as e:
            logger.error(f"Failed to load vector search engine: {e}")
            self.vector_engine = None
        
    def load_datasets(self):
        """Load datasets from configuration and external sources"""
        try:
            self.status_label.setText("Loading datasets...")
            
            # Load registered datasets
            registered = self.config_manager.get_registered_datasets()
            logger.info(f"Loaded {len(registered)} registered datasets")
            
            # Discover datasets from configured database connections
            self.status_label.setText("Discovering external datasets...")
            discovered_datasets = self.discover_database_datasets()
            logger.info(f"Discovered {len(discovered_datasets)} external datasets")
            
            # Discover local database datasets
            self.status_label.setText("Discovering local datasets...")
            local_datasets = self.discover_local_datasets()
            logger.info(f"Discovered {len(local_datasets)} local datasets")
            
            # Combine all datasets
            self.all_datasets = registered + discovered_datasets + local_datasets
            
            # Auto-index new datasets to vector database
            if discovered_datasets or local_datasets:
                self.status_label.setText("Indexing new datasets...")
                self.auto_index_datasets(discovered_datasets + local_datasets)
            
            self.apply_filters()
            
            logger.info(f"Total datasets loaded: {len(self.all_datasets)}")
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            self.status_label.setText(f"Error loading datasets: {str(e)}")
            self.all_datasets = []
            self.apply_filters()
    
    def apply_dataset_filters(self, filters: Dict[str, Any]):
        """Apply filters to the dataset list"""
        self.current_filters = filters
        self.apply_filters()
        
    def apply_filters(self):
        """Apply current filters to dataset list"""
        search_text = self.search_input.text().lower()
        type_filter = self.type_filter.currentText()
        access_filter = self.access_filter.currentText()
        
        # Get additional filters from filter dialog
        name_filter = self.current_filters.get('name_contains', '').lower()
        owner_filter = self.current_filters.get('owner_contains', '').lower()
        allowed_types = self.current_filters.get('types', [])
        allowed_access = self.current_filters.get('access_levels', [])
        include_tags = self.current_filters.get('include_tags', [])
        exclude_tags = self.current_filters.get('exclude_tags', [])
        min_quality = self.current_filters.get('min_quality_score', 0)
        supports_chat = self.current_filters.get('supports_chat', None)
        has_embeddings = self.current_filters.get('has_embeddings', None)
        has_documentation = self.current_filters.get('has_documentation', None)
        
        self.filtered_datasets = []
        
        for dataset in self.all_datasets:
            # Text search (existing)
            if search_text:
                searchable_text = f"{dataset.get('name', '')} {dataset.get('description', '')} {dataset.get('owner', '')}".lower()
                if search_text not in searchable_text:
                    continue
                    
            # Type filter (existing)
            if type_filter != "All Types":
                if dataset.get('type', '') != type_filter:
                    continue
                    
            # Access filter (existing)
            if access_filter != "All Access":
                access_level = dataset.get('access_level', '')
                if access_filter == "Full Access" and access_level != "Full":
                    continue
                elif access_filter == "Read Access" and access_level not in ["Read", "Read-Only"]:
                    continue
                elif access_filter == "No Access" and access_level != "No Access":
                    continue
            
            # Advanced filters from filter dialog
            
            # Name filter
            if name_filter and name_filter not in dataset.get('name', '').lower():
                continue
                
            # Owner filter
            if owner_filter and owner_filter not in dataset.get('owner', '').lower():
                continue
                
            # Dataset type filter (from advanced dialog)
            if allowed_types and dataset.get('type', '') not in allowed_types:
                continue
                
            # Access level filter (from advanced dialog)
            if allowed_access and dataset.get('access_level', '') not in allowed_access:
                continue
                
            # Tag filters
            dataset_tags = dataset.get('tags', [])
            if include_tags:
                if not any(tag in dataset_tags for tag in include_tags):
                    continue
                    
            if exclude_tags:
                if any(tag in dataset_tags for tag in exclude_tags):
                    continue
                    
            # Quality filter
            dataset_quality = dataset.get('quality_score', 0)
            if dataset_quality < min_quality:
                continue
                
            # AI capabilities filters
            if supports_chat is not None:
                if dataset.get('supports_chat', False) != supports_chat:
                    continue
                    
            if has_embeddings is not None:
                if dataset.get('has_embeddings', False) != has_embeddings:
                    continue
                    
            if has_documentation is not None:
                if dataset.get('has_documentation', False) != has_documentation:
                    continue
                    
            self.filtered_datasets.append(dataset)
            
        self.display_datasets()

    def display_datasets(self):
        """Display filtered datasets"""
        # Clear existing items
        while self.datasets_layout.count():
            child = self.datasets_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Add filtered datasets
        for dataset in self.filtered_datasets:
            dataset_item = DatasetItem(dataset)
            dataset_item.dataset_selected.connect(self.dataset_selected.emit)
            dataset_item.chat_requested.connect(self.chat_requested.emit)
            dataset_item.access_requested.connect(self.handle_access_request)
            self.datasets_layout.addWidget(dataset_item)
            
        # Add stretch to push items to top
        self.datasets_layout.addStretch()
        
        # Update status
        total_count = len(self.all_datasets)
        filtered_count = len(self.filtered_datasets)
        
        if filtered_count != total_count:
            self.status_label.setText(f"Showing {filtered_count} of {total_count} datasets")
        else:
            self.status_label.setText(f"Found {total_count} datasets")
            
    def handle_access_request(self, dataset_info: Dict[str, Any]):
        """Handle access request for a dataset"""
        dialog = AccessRequestDialog(dataset_info, self)
        if dialog.exec():
            # In a real implementation, this would send the request
            QMessageBox.information(
                self, 
                "Request Sent", 
                f"Access request for '{dataset_info['name']}' has been sent to {dataset_info['owner']}."
            )
            
    def discover_database_datasets(self) -> List[Dict[str, Any]]:
        """Discover datasets from configured database connections"""
        datasets = []
        
        try:
            # Get database connections from config
            db_connections = self.config_manager.get("database.connections", {})
            
            for conn_name, conn_config in db_connections.items():
                if not conn_config.get('enabled', True):
                    continue
                    
                try:
                    db_datasets = self._discover_from_connection(conn_name, conn_config)
                    datasets.extend(db_datasets)
                except Exception as e:
                    logger.error(f"Error discovering datasets from {conn_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error discovering database datasets: {e}")
            
        return datasets
    
    def discover_local_datasets(self) -> List[Dict[str, Any]]:
        """Discover datasets from local SQLite and DuckDB databases"""
        datasets = []
        
        try:
            # Check for local SQLite databases
            local_db_path = self.config_manager.get("local_database.sqlite_path", "data/local.db")
            if Path(local_db_path).exists():
                sqlite_datasets = self._discover_sqlite_datasets(local_db_path)
                datasets.extend(sqlite_datasets)
            
            # Check for local DuckDB databases
            duckdb_path = self.config_manager.get("local_database.duckdb_path", "data/local.duckdb")
            if Path(duckdb_path).exists():
                duckdb_datasets = self._discover_duckdb_datasets(duckdb_path)
                datasets.extend(duckdb_datasets)
                
        except Exception as e:
            logger.error(f"Error discovering local datasets: {e}")
            
        return datasets
    
    def _discover_from_connection(self, conn_name: str, conn_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover datasets from a specific database connection"""
        datasets = []
        db_type = conn_config.get('type', '').lower()
        
        try:
            if db_type == 'postgresql':
                datasets = self._discover_postgresql_datasets(conn_name, conn_config)
            elif db_type in ['mssql', 'sqlserver']:
                datasets = self._discover_mssql_datasets(conn_name, conn_config)
            elif db_type == 'bigquery':
                datasets = self._discover_bigquery_datasets(conn_name, conn_config)
            elif db_type == 'azure_sql':
                datasets = self._discover_azure_sql_datasets(conn_name, conn_config)
            else:
                logger.warning(f"Unsupported database type: {db_type}")
                
        except Exception as e:
            logger.error(f"Error discovering datasets from {conn_name} ({db_type}): {e}")
            
        return datasets
    
    def _discover_sqlite_datasets(self, db_path: str) -> List[Dict[str, Any]]:
        """Discover tables and views from SQLite database"""
        datasets = []
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
                    row_count = cursor.fetchone()[0]
                    
                    # Get column info
                    cursor.execute(f"PRAGMA table_info([{table_name}])")
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    
                    datasets.append({
                        'name': f"SQLite.{table_name}",
                        'type': 'Table',
                        'description': f'SQLite table with {len(column_names)} columns: {", ".join(column_names[:5])}{"..." if len(column_names) > 5 else ""}',
                        'owner': 'Local Database',
                        'access_level': 'Full',
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'row_count': row_count,
                        'source': f'sqlite://{db_path}#{table_name}',
                        'supports_chat': True,
                        'tags': ['local', 'sqlite', 'table'],
                        'connection_name': 'local_sqlite',
                        'schema_info': f'Columns: {", ".join(column_names)}'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error analyzing SQLite table {table_name}: {e}")
            
            # Get views
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
            views = cursor.fetchall()
            
            for (view_name,) in views:
                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM [{view_name}]")
                    row_count = cursor.fetchone()[0]
                    
                    datasets.append({
                        'name': f"SQLite.{view_name}",
                        'type': 'View',
                        'description': f'SQLite view with {row_count} rows',
                        'owner': 'Local Database',
                        'access_level': 'Full',
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'row_count': row_count,
                        'source': f'sqlite://{db_path}#{view_name}',
                        'supports_chat': True,
                        'tags': ['local', 'sqlite', 'view'],
                        'connection_name': 'local_sqlite'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error analyzing SQLite view {view_name}: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error discovering SQLite datasets from {db_path}: {e}")
            
        return datasets
    
    def _discover_duckdb_datasets(self, db_path: str) -> List[Dict[str, Any]]:
        """Discover tables and views from DuckDB database"""
        datasets = []
        
        try:
            import duckdb
            
            conn = duckdb.connect(db_path)
            
            # Get tables
            tables_result = conn.execute("SHOW TABLES").fetchall()
            
            for (table_name,) in tables_result:
                try:
                    # Get row count
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    
                    # Get column info
                    columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
                    column_names = [col[0] for col in columns_result]
                    
                    datasets.append({
                        'name': f"DuckDB.{table_name}",
                        'type': 'Table',
                        'description': f'DuckDB table with {len(column_names)} columns: {", ".join(column_names[:5])}{"..." if len(column_names) > 5 else ""}',
                        'owner': 'Local Database',
                        'access_level': 'Full',
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'row_count': row_count,
                        'source': f'duckdb://{db_path}#{table_name}',
                        'supports_chat': True,
                        'tags': ['local', 'duckdb', 'table'],
                        'connection_name': 'local_duckdb',
                        'schema_info': f'Columns: {", ".join(column_names)}'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error analyzing DuckDB table {table_name}: {e}")
            
            conn.close()
            
        except ImportError:
            logger.warning("DuckDB not available - install with: pip install duckdb")
        except Exception as e:
            logger.error(f"Error discovering DuckDB datasets from {db_path}: {e}")
            
        return datasets
    
    def _discover_postgresql_datasets(self, conn_name: str, conn_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover datasets from PostgreSQL database"""
        datasets = []
        
        if not self.connection_manager:
            logger.warning("Connection manager not available")
            return datasets
        
        try:
            conn = self.connection_manager.get_connection(conn_name)
            cursor = conn.cursor()
            
            # Get tables and views
            cursor.execute("""
                SELECT table_name, table_type, table_schema
                FROM information_schema.tables 
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_schema, table_name
            """)
            
            tables = cursor.fetchall()
            
            for table_name, table_type, schema_name in tables:
                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    # Get column info
                    cursor.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                    """, (schema_name, table_name))
                    
                    columns = cursor.fetchall()
                    column_info = [f"{col[0]} ({col[1]})" for col in columns]
                    
                    datasets.append({
                        'name': f"{conn_name}.{schema_name}.{table_name}",
                        'type': 'Table' if table_type == 'BASE TABLE' else 'View',
                        'description': f'PostgreSQL {table_type.lower()} with {len(columns)} columns',
                        'owner': conn_config.get('username', 'Unknown'),
                        'access_level': 'Full',
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'row_count': row_count,
                        'source': f'postgresql://{conn_config["host"]}:{conn_config.get("port", 5432)}/{conn_config["database"]}#{schema_name}.{table_name}',
                        'supports_chat': True,
                        'tags': ['postgresql', 'external', schema_name],
                        'connection_name': conn_name,
                        'schema_info': f'Columns: {", ".join([col[0] for col in columns])}'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error analyzing PostgreSQL table {schema_name}.{table_name}: {e}")
            
            conn.close()
            
        except ImportError:
            logger.warning("PostgreSQL driver not available - install with: pip install psycopg2-binary")
        except Exception as e:
            logger.error(f"Error discovering PostgreSQL datasets from {conn_name}: {e}")
            
        return datasets
    
    def _discover_mssql_datasets(self, conn_name: str, conn_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover datasets from MSSQL/SQL Server database"""
        datasets = []
        
        if not self.connection_manager:
            logger.warning("Connection manager not available")
            return datasets
        
        try:
            conn = self.connection_manager.get_connection(conn_name)
            cursor = conn.cursor()
            
            # Get tables and views
            cursor.execute("""
                SELECT TABLE_NAME, TABLE_TYPE, TABLE_SCHEMA
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')
                ORDER BY TABLE_SCHEMA, TABLE_NAME
            """)
            
            tables = cursor.fetchall()
            
            for table_name, table_type, schema_name in tables:
                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM [{schema_name}].[{table_name}]")
                    row_count = cursor.fetchone()[0]
                    
                    # Get column info
                    cursor.execute("""
                        SELECT COLUMN_NAME, DATA_TYPE 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                        ORDER BY ORDINAL_POSITION
                    """, schema_name, table_name)
                    
                    columns = cursor.fetchall()
                    
                    datasets.append({
                        'name': f"{conn_name}.{schema_name}.{table_name}",
                        'type': 'Table' if table_type == 'BASE TABLE' else 'View',
                        'description': f'SQL Server {table_type.lower()} with {len(columns)} columns',
                        'owner': conn_config.get('username', 'Unknown'),
                        'access_level': 'Full',
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'row_count': row_count,
                        'source': f'mssql://{conn_config["host"]}/{conn_config["database"]}#{schema_name}.{table_name}',
                        'supports_chat': True,
                        'tags': ['mssql', 'external', schema_name],
                        'connection_name': conn_name,
                        'schema_info': f'Columns: {", ".join([col[0] for col in columns])}'
                    })
                    
                except Exception as e:
                    logger.warning(f"Error analyzing SQL Server table {schema_name}.{table_name}: {e}")
            
            conn.close()
            
        except ImportError:
            logger.warning("SQL Server driver not available - install with: pip install pyodbc")
        except Exception as e:
            logger.error(f"Error discovering SQL Server datasets from {conn_name}: {e}")
            
        return datasets
    
    def _discover_bigquery_datasets(self, conn_name: str, conn_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover datasets from BigQuery"""
        datasets = []
        
        if not self.connection_manager:
            logger.warning("Connection manager not available")
            return datasets
        
        try:
            client = self.connection_manager.get_connection(conn_name)
            
            # List datasets
            bq_datasets = list(client.list_datasets())
            
            for dataset in bq_datasets:
                try:
                    # List tables in dataset
                    tables = list(client.list_tables(dataset.dataset_id))
                    
                    for table in tables:
                        try:
                            # Get table details
                            table_ref = client.get_table(table.reference)
                            
                            datasets.append({
                                'name': f"{conn_name}.{dataset.dataset_id}.{table.table_id}",
                                'type': 'Table' if table_ref.table_type == 'TABLE' else 'View',
                                'description': f'BigQuery {table_ref.table_type.lower()} with {len(table_ref.schema)} columns',
                                'owner': conn_config.get('project_id', 'Unknown'),
                                'access_level': 'Read',
                                'last_updated': table_ref.modified.strftime('%Y-%m-%d') if table_ref.modified else 'Unknown',
                                'row_count': table_ref.num_rows or 0,
                                'source': f'bigquery://{conn_config["project_id"]}/{dataset.dataset_id}/{table.table_id}',
                                'supports_chat': True,
                                'tags': ['bigquery', 'external', dataset.dataset_id],
                                'connection_name': conn_name,
                                'schema_info': f'Columns: {", ".join([field.name for field in table_ref.schema])}'
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error analyzing BigQuery table {table.table_id}: {e}")
                            
                except Exception as e:
                    logger.warning(f"Error listing tables in BigQuery dataset {dataset.dataset_id}: {e}")
            
        except ImportError:
            logger.warning("BigQuery client not available - install with: pip install google-cloud-bigquery")
        except Exception as e:
            logger.error(f"Error discovering BigQuery datasets from {conn_name}: {e}")
            
        return datasets
    
    def _discover_azure_sql_datasets(self, conn_name: str, conn_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover datasets from Azure SQL Database"""
        # Azure SQL uses the same protocol as SQL Server, so we can reuse the MSSQL discovery
        return self._discover_mssql_datasets(conn_name, conn_config)
    
    def auto_index_datasets(self, datasets):
        """Automatically index new datasets to vector database with duplicate prevention"""
        if not self.vector_engine:
            return
            
        for dataset in datasets:
            try:
                # Create virtual document for dataset
                dataset_id = f"dataset://{dataset['name']}"
                
                # Index the dataset as a virtual document (vector engine handles duplicates)
                success = self.vector_engine.index_document(
                    dataset_id,
                    fileset_name=dataset['name'],
                    fileset_description=dataset.get('description', ''),
                    tags=dataset.get('tags', []),
                    user_description=f"""
Dataset: {dataset['name']}
Type: {dataset.get('type', 'Unknown')}
Description: {dataset.get('description', '')}
Owner: {dataset.get('owner', 'Unknown')}
Source: {dataset.get('source', '')}
Access Level: {dataset.get('access_level', 'Unknown')}
Row Count: {dataset.get('row_count', 'Unknown')}
Schema: {dataset.get('schema_info', 'Unknown')}
Connection: {dataset.get('connection_name', 'Unknown')}
                    """.strip()
                )
                
                if success:
                    logger.info(f"Auto-indexed dataset: {dataset['name']}")
                else:
                    logger.warning(f"Failed to auto-index dataset: {dataset['name']}")
                        
            except Exception as e:
                logger.error(f"Error auto-indexing dataset {dataset.get('name', 'Unknown')}: {e}")
    
    def refresh(self):
        """Refresh dataset list"""
        try:
            self.status_label.setText("Refreshing datasets...")
            
            # Clear existing datasets
            self.all_datasets = []
            self.filtered_datasets = []
            
            # Reload datasets
            self.load_datasets()
            
            self.status_label.setText(f"Refreshed - Found {len(self.all_datasets)} datasets")
            
        except Exception as e:
            logger.error(f"Error refreshing datasets: {e}")
            self.status_label.setText(f"Refresh failed: {str(e)}")
        
    def get_selected_datasets(self) -> List[Dict[str, Any]]:
        """Get currently filtered/selected datasets"""
        return self.filtered_datasets.copy()

class AccessRequestDialog(QDialog):
    """Dialog for requesting access to a dataset"""
    
    def __init__(self, dataset_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.dataset_info = dataset_info
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the access request dialog"""
        self.setWindowTitle("Request Dataset Access")
        self.setModal(True)
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Dataset info
        info_label = QLabel(f"<h3>Request access to: {self.dataset_info['name']}</h3>")
        layout.addWidget(info_label)
        
        owner_label = QLabel(f"<b>Owner:</b> {self.dataset_info['owner']}")
        layout.addWidget(owner_label)
        
        desc_label = QLabel(f"<b>Description:</b> {self.dataset_info['description']}")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Access level requested
        form_layout = QFormLayout()
        
        self.access_level = QComboBox()
        self.access_level.addItems(["Read-Only", "Read-Write", "Full Access"])
        form_layout.addRow("Access Level:", self.access_level)
        
        # Business justification
        self.justification = QTextEdit()
        self.justification.setPlaceholderText("Please explain why you need access to this dataset...")
        self.justification.setMaximumHeight(100)
        form_layout.addRow("Justification:", self.justification)
        
        # Duration
        self.duration = QComboBox()
        self.duration.addItems(["30 days", "90 days", "6 months", "1 year", "Permanent"])
        form_layout.addRow("Duration:", self.duration)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        
        request_btn = QPushButton("Send Request")
        request_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        request_btn.clicked.connect(self.accept)
        button_layout.addWidget(request_btn)
        
        layout.addWidget(QFrame())  # Spacer
        layout.addLayout(button_layout)
        
    def get_request_data(self) -> Dict[str, Any]:
        """Get the access request data"""
        return {
            'dataset': self.dataset_info,
            'access_level': self.access_level.currentText(),
            'justification': self.justification.toPlainText(),
            'duration': self.duration.currentText()
        }
