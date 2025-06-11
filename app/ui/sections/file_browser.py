"""
File Browser Widget
Shows files in watched directories with preview and management capabilities
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QLabel, 
    QPushButton, QLineEdit, QComboBox, QMenu, QMessageBox,
    QFileDialog, QFrame, QProgressBar, QDialog, QFormLayout, QTextEdit,
    QListWidget, QListWidgetItem, QSplitter, QGroupBox, QCheckBox, QTabWidget
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QModelIndex, QThread, QFileSystemWatcher
)
from PyQt6.QtGui import QFont, QAction, QStandardItemModel, QStandardItem, QIcon, QPixmap, QPainter, QColor
from typing import Dict, Any

import logging

logger = logging.getLogger(__name__)

class FileIndexWorker(QThread):
    """Background worker for indexing files"""
    
    progress_updated = pyqtSignal(int, str)  # progress, current_file
    file_indexed = pyqtSignal(dict)  # file_info
    indexing_complete = pyqtSignal(int)  # total_files_indexed
    
    def __init__(self, directories: List[str], supported_formats: List[str]):
        super().__init__()
        self.directories = directories
        self.supported_formats = supported_formats
        self.should_stop = False
        
    def run(self):
        """Index files in background"""
        total_files = 0
        indexed_files = 0
        
        try:
            # First pass: count total files
            for directory in self.directories:
                if self.should_stop:
                    return
                    
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if any(file.lower().endswith(fmt.lower()) for fmt in self.supported_formats):
                            total_files += 1
            
            # Second pass: index files
            for directory in self.directories:
                if self.should_stop:
                    return
                    
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if self.should_stop:
                            return
                            
                        if any(file.lower().endswith(fmt.lower()) for fmt in self.supported_formats):
                            file_path = Path(root) / file
                            file_info = self.get_file_info(file_path)
                            
                            self.file_indexed.emit(file_info)
                            indexed_files += 1
                            
                            progress = int((indexed_files / total_files) * 100) if total_files > 0 else 0
                            self.progress_updated.emit(progress, str(file_path))
            
            self.indexing_complete.emit(indexed_files)
            
        except Exception as e:
            logger.error(f"Error during file indexing: {e}")
            
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information"""
        try:
            stat = file_path.stat()
            return {
                'name': file_path.name,
                'path': str(file_path),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'type': file_path.suffix.lower(),
                'directory': str(file_path.parent),
                'is_accessible': os.access(file_path, os.R_OK),
                'description': self.get_file_description(file_path)
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {
                'name': file_path.name,
                'path': str(file_path),
                'size': 0,
                'modified': 0,
                'type': file_path.suffix.lower(),
                'directory': str(file_path.parent),
                'is_accessible': False,
                'description': f"Error: {str(e)}"
            }
            
    def get_file_description(self, file_path: Path) -> str:
        """Get basic file description"""
        file_type = file_path.suffix.lower()
        
        type_descriptions = {
            '.pdf': 'PDF Document',
            '.docx': 'Word Document',
            '.xlsx': 'Excel Spreadsheet',
            '.csv': 'CSV Data File',
            '.json': 'JSON Data File',
            '.txt': 'Text File',
            '.md': 'Markdown Document',
            '.py': 'Python Script',
            '.sql': 'SQL Script'
        }
        
        return type_descriptions.get(file_type, f'{file_type.upper()} File')
        
    def stop(self):
        """Stop the indexing process"""
        self.should_stop = True

class FileMetadataManager:
    """Manages individual file metadata storage and retrieval"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.metadata_cache = {}
        self.load_metadata()
        
    def load_metadata(self):
        """Load file metadata from config"""
        self.metadata_cache = self.config_manager.get("file_management.file_metadata", {})
        
    def save_metadata(self):
        """Save file metadata to config"""
        self.config_manager.set("file_management.file_metadata", self.metadata_cache)
        
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a specific file"""
        return self.metadata_cache.get(file_path, {
            'tags': [],
            'description': '',
            'category': '',
            'priority': 'Normal',
            'custom_fields': {},
            'indexed_for_vector_search': False,
            'last_metadata_update': ''
        })
        
    def set_file_metadata(self, file_path: str, metadata: Dict[str, Any]):
        """Set metadata for a specific file"""
        from datetime import datetime
        metadata['last_metadata_update'] = datetime.now().isoformat()
        self.metadata_cache[file_path] = metadata
        self.save_metadata()
        
    def remove_file_metadata(self, file_path: str):
        """Remove metadata for a file"""
        if file_path in self.metadata_cache:
            del self.metadata_cache[file_path]
            self.save_metadata()

class FileBrowser(QWidget):
    """File browser widget for showing and managing files with folder-like interface"""
    
    file_selected = pyqtSignal(str)  # file_path
    file_double_clicked = pyqtSignal(str)  # file_path
    directory_added = pyqtSignal(str)  # directory_path
    file_metadata_changed = pyqtSignal(str, dict)  # file_path, metadata
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.metadata_manager = FileMetadataManager(config_manager)
        self.indexed_files = []
        self.current_filter = ""
        self.indexer_worker = None
        self.file_watcher = QFileSystemWatcher()
        self.current_directory = None
        
        self.setup_ui()
        self.setup_connections()
        self.load_watched_directories()
        
    def setup_ui(self):
        """Setup the file browser UI with folder-like interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with controls
        header_layout = QVBoxLayout()
        
        # Title and controls
        title_layout = QHBoxLayout()
        
        title_label = QLabel("Files")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # View mode toggle
        self.view_mode = QComboBox()
        self.view_mode.addItems(["List View", "Folder View"])
        self.view_mode.setCurrentText("Folder View")
        self.view_mode.currentTextChanged.connect(self.change_view_mode)
        title_layout.addWidget(self.view_mode)
        
        add_dir_button = QPushButton("+ Add Directory")
        add_dir_button.setToolTip("Add directory to watch")
        add_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        add_dir_button.clicked.connect(self.add_directory)
        title_layout.addWidget(add_dir_button)
        
        header_layout.addLayout(title_layout)
        
        # Navigation breadcrumb
        self.breadcrumb_layout = QHBoxLayout()
        self.breadcrumb_frame = QFrame()
        self.breadcrumb_frame.setLayout(self.breadcrumb_layout)
        self.breadcrumb_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        header_layout.addWidget(self.breadcrumb_frame)
        
        # Search and filter
        filter_layout = QHBoxLayout()
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search files...")
        self.filter_input.setStyleSheet("""
            QLineEdit {
                padding: 4px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 9pt;
            }
        """)
        filter_layout.addWidget(self.filter_input)
        
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All Types", ".pdf", ".docx", ".xlsx", ".csv", ".json", ".txt", ".py", ".md"])
        self.type_filter.setStyleSheet("font-size: 9pt;")
        filter_layout.addWidget(self.type_filter)
        
        self.metadata_filter = QComboBox()
        self.metadata_filter.addItems(["All Files", "With Metadata", "Without Metadata", "Vector Indexed"])
        self.metadata_filter.setStyleSheet("font-size: 9pt;")
        filter_layout.addWidget(self.metadata_filter)
        
        header_layout.addLayout(filter_layout)
        
        layout.addLayout(header_layout)
        
        # Indexing progress
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Indexing files...")
        self.progress_label.setStyleSheet("font-size: 8pt; color: #666;")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(self.progress_frame)
        
        # Main content area with splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Directory tree and file list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Directory tree
        self.directory_tree = QTreeView()
        self.directory_model = QStandardItemModel()
        self.directory_model.setHorizontalHeaderLabels(['Watched Directories'])
        self.directory_tree.setModel(self.directory_model)
        self.directory_tree.setMaximumHeight(150)
        self.directory_tree.setStyleSheet("""
            QTreeView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
        """)
        left_layout.addWidget(self.directory_tree)
        
        # File list/grid
        self.file_container = QWidget()
        self.setup_file_views()
        left_layout.addWidget(self.file_container)
        
        main_splitter.addWidget(left_widget)
        
        # Right side: File details and metadata
        self.details_panel = self.create_details_panel()
        main_splitter.addWidget(self.details_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 300])
        
        layout.addWidget(main_splitter)
        
        # Status label
        self.status_label = QLabel("No directories being watched")
        self.status_label.setStyleSheet("font-size: 8pt; color: #666; padding: 4px;")
        layout.addWidget(self.status_label)
        
    def setup_file_views(self):
        """Setup both list and folder view for files"""
        container_layout = QVBoxLayout(self.file_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # List view (tree view)
        self.file_model = QStandardItemModel()
        self.file_model.setHorizontalHeaderLabels(['Name', 'Type', 'Size', 'Tags'])
        
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setSortingEnabled(True)
        self.file_tree.setRootIsDecorated(False)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.setStyleSheet("""
            QTreeView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #e3f2fd;
            }
            QTreeView::item {
                padding: 6px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTreeView::item:hover {
                background-color: #f5f5f5;
            }
            QTreeView::item:selected {
                background-color: #e3f2fd;
                color: black;
            }
        """)
        
        # Set column widths
        self.file_tree.setColumnWidth(0, 200)  # Name
        self.file_tree.setColumnWidth(1, 60)   # Type
        self.file_tree.setColumnWidth(2, 80)   # Size
        self.file_tree.setColumnWidth(3, 120)  # Tags
        
        # Folder view (list widget with icons)
        self.file_list = QListWidget()
        self.file_list.setViewMode(QListWidget.ViewMode.IconMode)
        from PyQt6.QtCore import QSize
        self.file_list.setIconSize(QSize(48, 48))
        self.file_list.setGridSize(QSize(80, 100))
        self.file_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #e3f2fd;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: black;
            }
        """)
        
        container_layout.addWidget(self.file_tree)
        container_layout.addWidget(self.file_list)
        
        # Initially show folder view
        self.file_tree.setVisible(False)
        self.file_list.setVisible(True)
        
    def create_details_panel(self):
        """Create the file details and metadata panel"""
        details_widget = QWidget()
        details_widget.setMaximumWidth(350)
        details_widget.setMinimumWidth(250)
        
        layout = QVBoxLayout(details_widget)
        
        # Panel title
        title_label = QLabel("File Details")
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setStyleSheet("padding: 8px; background-color: #f8f9fa; border-radius: 4px;")
        layout.addWidget(title_label)
        
        # Tabs for different detail views
        self.details_tabs = QTabWidget()
        
        # Properties tab
        self.properties_tab = self.create_properties_tab()
        self.details_tabs.addTab(self.properties_tab, "Properties")
        
        # Metadata tab
        self.metadata_tab = self.create_metadata_tab()
        self.details_tabs.addTab(self.metadata_tab, "Metadata")
        
        # Preview tab
        self.preview_tab = self.create_preview_tab()
        self.details_tabs.addTab(self.preview_tab, "Preview")
        
        layout.addWidget(self.details_tabs)
        
        # No file selected message
        self.no_selection_label = QLabel("Select a file to view details")
        self.no_selection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_selection_label.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
        layout.addWidget(self.no_selection_label)
        
        # Initially hide tabs
        self.details_tabs.setVisible(False)
        
        return details_widget
        
    def create_properties_tab(self):
        """Create the file properties tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File info labels
        self.file_name_label = QLabel()
        self.file_name_label.setWordWrap(True)
        self.file_name_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.file_name_label)
        
        self.file_path_label = QLabel()
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet("color: #666; font-size: 8pt; padding: 4px;")
        layout.addWidget(self.file_path_label)
        
        # Properties group
        props_group = QGroupBox("Properties")
        props_layout = QFormLayout(props_group)
        
        self.file_size_label = QLabel()
        props_layout.addRow("Size:", self.file_size_label)
        
        self.file_type_label = QLabel()
        props_layout.addRow("Type:", self.file_type_label)
        
        self.file_modified_label = QLabel()
        props_layout.addRow("Modified:", self.file_modified_label)
        
        self.file_accessible_label = QLabel()
        props_layout.addRow("Accessible:", self.file_accessible_label)
        
        layout.addWidget(props_group)
        
        layout.addStretch()
        return widget
        
    def create_metadata_tab(self):
        """Create the file metadata tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Metadata form
        form_layout = QFormLayout()
        
        # Tags
        self.metadata_tags = QLineEdit()
        self.metadata_tags.setPlaceholderText("tag1, tag2, tag3")
        form_layout.addRow("Tags:", self.metadata_tags)
        
        # Description
        self.metadata_description = QTextEdit()
        self.metadata_description.setMaximumHeight(80)
        self.metadata_description.setPlaceholderText("File description...")
        form_layout.addRow("Description:", self.metadata_description)
        
        # Category
        self.metadata_category = QComboBox()
        self.metadata_category.addItems(["", "Document", "Data", "Code", "Image", "Archive", "Other"])
        self.metadata_category.setEditable(True)
        form_layout.addRow("Category:", self.metadata_category)
        
        # Priority
        self.metadata_priority = QComboBox()
        self.metadata_priority.addItems(["Low", "Normal", "High", "Critical"])
        self.metadata_priority.setCurrentText("Normal")
        form_layout.addRow("Priority:", self.metadata_priority)
        
        # Vector search indexing
        self.metadata_vector_indexed = QCheckBox("Include in vector search")
        self.metadata_vector_indexed.setChecked(True)
        form_layout.addRow(self.metadata_vector_indexed)
        
        layout.addLayout(form_layout)
        
        # Save button
        save_metadata_btn = QPushButton("Save Metadata")
        save_metadata_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0056b3; }
        """)
        save_metadata_btn.clicked.connect(self.save_file_metadata)
        layout.addWidget(save_metadata_btn)
        
        layout.addStretch()
        return widget
        
    def create_preview_tab(self):
        """Create the file preview tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.preview_content = QTextEdit()
        self.preview_content.setReadOnly(True)
        self.preview_content.setPlaceholderText("File preview will appear here...")
        self.preview_content.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self.preview_content)
        
        return widget
        
    def change_view_mode(self, mode):
        """Change between list and folder view"""
        if mode == "List View":
            self.file_tree.setVisible(True)
            self.file_list.setVisible(False)
        else:  # Folder View
            self.file_tree.setVisible(False)
            self.file_list.setVisible(True)
            
    def update_breadcrumb(self, path=None):
        """Update the breadcrumb navigation"""
        # Clear existing breadcrumb
        while self.breadcrumb_layout.count():
            child = self.breadcrumb_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        if path:
            from pathlib import Path
            path_obj = Path(path)
            
            # Add home button
            home_btn = QPushButton("📁 Watched Directories")
            home_btn.setFlat(True)
            home_btn.clicked.connect(lambda: self.navigate_to_directory(None))
            self.breadcrumb_layout.addWidget(home_btn)
            
            # Add separator
            sep_label = QLabel(" > ")
            sep_label.setStyleSheet("color: #666;")
            self.breadcrumb_layout.addWidget(sep_label)
            
            # Add current directory
            dir_btn = QPushButton(f"📂 {path_obj.name}")
            dir_btn.setFlat(True)
            dir_btn.setStyleSheet("font-weight: bold;")
            self.breadcrumb_layout.addWidget(dir_btn)
            
        else:
            # Show watched directories
            home_label = QLabel("📁 All Watched Directories")
            home_label.setStyleSheet("font-weight: bold; color: #333;")
            self.breadcrumb_layout.addWidget(home_label)
            
        self.breadcrumb_layout.addStretch()
        
    def navigate_to_directory(self, directory_path):
        """Navigate to a specific directory"""
        self.current_directory = directory_path
        self.update_breadcrumb(directory_path)
        self.apply_filters()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.filter_input.textChanged.connect(self.apply_filters)
        self.type_filter.currentTextChanged.connect(self.apply_filters)
        self.metadata_filter.currentTextChanged.connect(self.apply_filters)

        # File selection connections
        self.file_tree.clicked.connect(self.on_file_clicked)
        self.file_tree.doubleClicked.connect(self.on_file_double_clicked)
        self.file_list.itemClicked.connect(self.on_file_list_clicked)
        self.file_list.itemDoubleClicked.connect(self.on_file_list_double_clicked)

        # Directory tree connections
        self.directory_tree.clicked.connect(self.on_directory_clicked)

        # Context menus
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_context_menu)

        # File watcher
        self.file_watcher.directoryChanged.connect(self.on_directory_changed)
        self.file_watcher.fileChanged.connect(self.on_file_changed)
        
    def load_watched_directories(self):
        """Load watched directories from config and populate directory tree"""
        watched_dirs = self.config_manager.get("file_management.watched_directories", [])
        
        # Clear directory model
        self.directory_model.clear()
        self.directory_model.setHorizontalHeaderLabels(['Watched Directories'])
        
        # Add watched directories to tree
        for directory in watched_dirs:
            if Path(directory).exists():
                dir_item = QStandardItem(f"📂 {Path(directory).name}")
                dir_item.setData(directory, Qt.ItemDataRole.UserRole)
                dir_item.setToolTip(directory)
                self.directory_model.appendRow(dir_item)
                
                # Add to file watcher
                self.file_watcher.addPath(directory)
        
        if watched_dirs:
            self.start_indexing()
            self.status_label.setText(f"Watching {len(watched_dirs)} directories")
        else:
            self.status_label.setText("No directories being watched")
            
        # Update breadcrumb
        self.update_breadcrumb()
            
    def add_directory(self):
        """Add directory to watch list with metadata collection"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Watch",
            str(Path.home())
        )
        
        if directory:
            # Show metadata dialog for the new directory
            metadata_dialog = DirectoryMetadataDialog(directory, self)
            if metadata_dialog.exec():
                metadata = metadata_dialog.get_metadata()
                
                # Add to watched directories
                self.config_manager.add_watched_directory(directory)
                self.directory_added.emit(directory)
                
                # Start indexing with metadata
                self.start_indexing_with_metadata(metadata)
            else:
                # User cancelled, add without metadata
                self.config_manager.add_watched_directory(directory)
                self.directory_added.emit(directory)
                self.start_indexing()
            
    def start_indexing(self):
        """Start indexing files in watched directories"""
        watched_dirs = self.config_manager.get("file_management.watched_directories", [])
        supported_formats = self.config_manager.get("file_management.supported_formats", [])
        
        if not watched_dirs:
            return
            
        # Stop existing indexer
        if self.indexer_worker and self.indexer_worker.isRunning():
            self.indexer_worker.stop()
            self.indexer_worker.wait()
            
        # Clear existing files
        self.file_model.clear()
        self.file_model.setHorizontalHeaderLabels(['Name', 'Type', 'Size'])
        self.indexed_files.clear()
        
        # Show progress
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting file indexing...")
        
        # Start new indexer
        self.indexer_worker = FileIndexWorker(watched_dirs, supported_formats)
        self.indexer_worker.progress_updated.connect(self.on_indexing_progress)
        self.indexer_worker.file_indexed.connect(self.on_file_indexed)
        self.indexer_worker.indexing_complete.connect(self.on_indexing_complete)
        self.indexer_worker.start()
        
        # Also trigger vector indexing
        self.trigger_vector_indexing()
        
    def start_indexing_with_metadata(self, metadata: Dict[str, Any]):
        """Start indexing with specific metadata for new directory"""
        # Store metadata for use during vector indexing
        self.pending_metadata = metadata
        self.start_indexing()
        
    def trigger_vector_indexing(self):
        """Trigger vector indexing of watched directories with metadata"""
        try:
            from utils.vector_search import VectorSearchEngine
            from PyQt6.QtCore import QThread, pyqtSignal
            
            # Create background worker for vector indexing
            class VectorIndexingWorker(QThread):
                progress_updated = pyqtSignal(str)  # status message
                indexing_complete = pyqtSignal(dict)  # results
                error_occurred = pyqtSignal(str)  # error message
                
                def __init__(self, config_manager, watched_dirs, metadata):
                    super().__init__()
                    self.config_manager = config_manager
                    self.watched_dirs = watched_dirs
                    self.metadata = metadata
                    
                def run(self):
                    try:
                        vector_engine = VectorSearchEngine(self.config_manager)
                        total_results = {
                            'total_directories': len(self.watched_dirs),
                            'indexed_directories': 0,
                            'total_files': 0,
                            'indexed_files': 0,
                            'errors': []
                        }
                        
                        for directory in self.watched_dirs:
                            if Path(directory).exists():
                                self.progress_updated.emit(f"Indexing directory: {Path(directory).name}")
                                
                                # Use pending metadata if available, otherwise auto-generate
                                if hasattr(self, 'metadata') and self.metadata:
                                    metadata = self.metadata
                                else:
                                    metadata = {
                                        'fileset_name': Path(directory).name,
                                        'description': f"Files from {directory}",
                                        'tags': ['files', 'local']
                                    }
                                
                                results = vector_engine.index_directory(
                                    directory,
                                    fileset_name=metadata.get('fileset_name'),
                                    fileset_description=metadata.get('description'),
                                    tags=metadata.get('tags', [])
                                )
                                
                                total_results['indexed_directories'] += 1
                                total_results['total_files'] += results.get('total_files', 0)
                                total_results['indexed_files'] += results.get('indexed_files', 0)
                                total_results['errors'].extend(results.get('errors', []))
                                
                        self.indexing_complete.emit(total_results)
                        
                    except Exception as e:
                        self.error_occurred.emit(str(e))
            
            vector_engine = VectorSearchEngine(self.config_manager)
            watched_dirs = self.config_manager.get("file_management.watched_directories", [])
            
            if watched_dirs:
                # Use pending metadata if available
                metadata = getattr(self, 'pending_metadata', {})
                
                # Start background vector indexing
                self.vector_worker = VectorIndexingWorker(self.config_manager, watched_dirs, metadata)
                self.vector_worker.progress_updated.connect(self.on_vector_progress)
                self.vector_worker.indexing_complete.connect(self.on_vector_complete)
                self.vector_worker.error_occurred.connect(self.on_vector_error)
                self.vector_worker.start()
                
                # Update status
                self.status_label.setText("Vector indexing in progress...")
                
            # Clear pending metadata
            if hasattr(self, 'pending_metadata'):
                delattr(self, 'pending_metadata')
                
        except Exception as e:
            logger.warning(f"Vector indexing failed: {e}")
            
    def on_vector_progress(self, status: str):
        """Handle vector indexing progress"""
        self.status_label.setText(f"Vector indexing: {status}")
        
    def on_vector_complete(self, results: Dict[str, Any]):
        """Handle vector indexing completion"""
        indexed_files = results.get('indexed_files', 0)
        total_files = results.get('total_files', 0)
        
        if indexed_files > 0:
            self.status_label.setText(f"Found {total_files} files, {indexed_files} indexed for vector search")
        else:
            self.status_label.setText(f"Found {total_files} files in watched directories")
            
    def on_vector_error(self, error: str):
        """Handle vector indexing error"""
        logger.error(f"Vector indexing error: {error}")
        # Don't show error to user unless it's critical, just log it
        
    def on_indexing_progress(self, progress: int, current_file: str):
        """Handle indexing progress updates"""
        self.progress_bar.setValue(progress)
        filename = Path(current_file).name
        self.progress_label.setText(f"Indexing: {filename}")
        
    def on_file_indexed(self, file_info: Dict[str, Any]):
        """Handle newly indexed file"""
        self.indexed_files.append(file_info)
        
        # Add to model if it passes current filter
        if self.passes_filter(file_info):
            self.add_file_to_model(file_info)
            
    def on_indexing_complete(self, total_files: int):
        """Handle indexing completion"""
        self.progress_frame.setVisible(False)
        self.status_label.setText(f"Found {total_files} files in watched directories")
        
        # Sort the tree
        self.file_tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        
    def add_file_to_model(self, file_info: Dict[str, Any]):
        """Add file to both tree and list models"""
        # Get file metadata
        metadata = self.metadata_manager.get_file_metadata(file_info['path'])
        
        # Create file icon based on type
        file_icon = self.get_file_icon(file_info['type'])
        
        # Add to tree view
        name_item = QStandardItem(file_info['name'])
        name_item.setData(file_info['path'], Qt.ItemDataRole.UserRole)
        name_item.setIcon(file_icon)
        
        # Enhanced tooltip with metadata
        tooltip_parts = [
            f"Path: {file_info['path']}",
            f"Type: {file_info['description']}",
            f"Size: {self.format_file_size(file_info['size'])}"
        ]
        if metadata.get('description'):
            tooltip_parts.append(f"Description: {metadata['description']}")
        if metadata.get('tags'):
            tooltip_parts.append(f"Tags: {', '.join(metadata['tags'])}")
        name_item.setToolTip('\n'.join(tooltip_parts))
        
        # Type item
        type_item = QStandardItem(file_info['type'])
        
        # Size item
        size_item = QStandardItem(self.format_file_size(file_info['size']))
        size_item.setData(file_info['size'], Qt.ItemDataRole.UserRole)  # Store actual size for sorting
        
        # Tags item
        tags_text = ', '.join(metadata.get('tags', []))
        tags_item = QStandardItem(tags_text)
        
        # Add accessibility indicator
        if not file_info['is_accessible']:
            name_item.setForeground(Qt.GlobalColor.red)
            name_item.setToolTip(f"{file_info['path']}\nAccess denied")
            
        self.file_model.appendRow([name_item, type_item, size_item, tags_item])
        
        # Add to list view
        list_item = QListWidgetItem()
        list_item.setText(file_info['name'])
        list_item.setIcon(file_icon)
        list_item.setData(Qt.ItemDataRole.UserRole, file_info['path'])
        list_item.setToolTip('\n'.join(tooltip_parts))
        
        # Add metadata indicators
        if metadata.get('tags'):
            list_item.setText(f"{file_info['name']}\n🏷️ {', '.join(metadata['tags'][:2])}")
        
        self.file_list.addItem(list_item)
        
    def get_file_icon(self, file_type: str):
        """Get icon for file type"""
        # Create simple colored icons for different file types
        icon_colors = {
            '.pdf': '#dc3545',    # Red
            '.docx': '#007bff',   # Blue
            '.xlsx': '#28a745',   # Green
            '.csv': '#28a745',    # Green
            '.json': '#ffc107',   # Yellow
            '.txt': '#6c757d',    # Gray
            '.py': '#17a2b8',     # Cyan
            '.md': '#6f42c1',     # Purple
            '.sql': '#fd7e14',    # Orange
        }
        
        color = icon_colors.get(file_type.lower(), '#6c757d')
        
        # Create a simple colored square icon
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor('#ffffff'))
        painter.setFont(QFont('Arial', 8, QFont.Weight.Bold))
        
        # Draw file extension
        ext_text = file_type[1:].upper() if file_type.startswith('.') else 'FILE'
        if len(ext_text) > 3:
            ext_text = ext_text[:3]
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, ext_text)
        painter.end()
        
        return QIcon(pixmap)
        
    def format_file_size(self, size: int) -> str:
        """Format file size in human readable format"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size // 1024} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size // (1024 * 1024)} MB"
        else:
            return f"{size // (1024 * 1024 * 1024)} GB"
        
    def passes_filter(self, file_info: Dict[str, Any]) -> bool:
        """Check if file passes current filters"""
        # Directory filter
        if self.current_directory:
            if not file_info['path'].startswith(self.current_directory):
                return False
        
        # Text filter
        if self.current_filter:
            search_text = f"{file_info['name']} {file_info['description']}".lower()
            metadata = self.metadata_manager.get_file_metadata(file_info['path'])
            if metadata.get('tags'):
                search_text += f" {' '.join(metadata['tags'])}"
            if metadata.get('description'):
                search_text += f" {metadata['description']}"
                
            if self.current_filter.lower() not in search_text:
                return False
                
        # Type filter
        type_filter = self.type_filter.currentText()
        if type_filter != "All Types" and file_info['type'] != type_filter:
            return False
            
        # Metadata filter
        metadata_filter = self.metadata_filter.currentText()
        if metadata_filter != "All Files":
            metadata = self.metadata_manager.get_file_metadata(file_info['path'])
            
            if metadata_filter == "With Metadata":
                if not (metadata.get('tags') or metadata.get('description') or metadata.get('category')):
                    return False
            elif metadata_filter == "Without Metadata":
                if metadata.get('tags') or metadata.get('description') or metadata.get('category'):
                    return False
            elif metadata_filter == "Vector Indexed":
                if not metadata.get('indexed_for_vector_search', False):
                    return False
            
        return True
        
    def apply_filters(self):
        """Apply current filters to file list"""
        self.current_filter = self.filter_input.text()
        
        # Clear and repopulate models
        self.file_model.clear()
        self.file_model.setHorizontalHeaderLabels(['Name', 'Type', 'Size', 'Tags'])
        self.file_list.clear()
        
        filtered_files = []
        for file_info in self.indexed_files:
            if self.passes_filter(file_info):
                self.add_file_to_model(file_info)
                filtered_files.append(file_info)
                
        # Update status
        visible_count = len(filtered_files)
        total_count = len(self.indexed_files)
        
        if self.current_directory:
            dir_name = Path(self.current_directory).name
            if visible_count != total_count:
                self.status_label.setText(f"Showing {visible_count} of {total_count} files in {dir_name}")
            else:
                self.status_label.setText(f"Found {total_count} files in {dir_name}")
        else:
            if visible_count != total_count:
                self.status_label.setText(f"Showing {visible_count} of {total_count} files")
            else:
                self.status_label.setText(f"Found {total_count} files in watched directories")
            
    def on_file_clicked(self, index: QModelIndex):
        """Handle file selection in tree view"""
        if index.isValid():
            name_item = self.file_model.item(index.row(), 0)
            file_path = name_item.data(Qt.ItemDataRole.UserRole)
            self.show_file_details(file_path)
            self.file_selected.emit(file_path)
            
    def on_file_double_clicked(self, index: QModelIndex):
        """Handle file double-click in tree view"""
        if index.isValid():
            name_item = self.file_model.item(index.row(), 0)
            file_path = name_item.data(Qt.ItemDataRole.UserRole)
            self.file_double_clicked.emit(file_path)
            
    def on_file_list_clicked(self, item: QListWidgetItem):
        """Handle file selection in list view"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.show_file_details(file_path)
        self.file_selected.emit(file_path)
        
    def on_file_list_double_clicked(self, item: QListWidgetItem):
        """Handle file double-click in list view"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.file_double_clicked.emit(file_path)
        
    def on_directory_clicked(self, index: QModelIndex):
        """Handle directory selection"""
        if index.isValid():
            item = self.directory_model.itemFromIndex(index)
            directory_path = item.data(Qt.ItemDataRole.UserRole)
            if directory_path:
                self.navigate_to_directory(directory_path)
                
    def on_directory_changed(self, path: str):
        """Handle directory changes from file watcher"""
        logger.info(f"Directory changed: {path}")
        # Trigger re-indexing of changed directory
        self.start_indexing()
        
    def on_file_changed(self, path: str):
        """Handle file changes from file watcher"""
        logger.info(f"File changed: {path}")
        # Update file info if it's in our list
        for i, file_info in enumerate(self.indexed_files):
            if file_info['path'] == path:
                # Re-index this specific file
                try:
                    updated_info = self.indexer_worker.get_file_info(Path(path))
                    self.indexed_files[i] = updated_info
                    self.apply_filters()
                except Exception as e:
                    logger.warning(f"Failed to update file info for {path}: {e}")
                break
                
    def show_file_details(self, file_path: str):
        """Show file details in the details panel"""
        if not file_path:
            self.details_tabs.setVisible(False)
            self.no_selection_label.setVisible(True)
            return
            
        # Find file info
        file_info = None
        for info in self.indexed_files:
            if info['path'] == file_path:
                file_info = info
                break
                
        if not file_info:
            return
            
        # Show details tabs
        self.details_tabs.setVisible(True)
        self.no_selection_label.setVisible(False)
        
        # Update properties tab
        self.file_name_label.setText(file_info['name'])
        self.file_path_label.setText(file_info['path'])
        self.file_size_label.setText(self.format_file_size(file_info['size']))
        self.file_type_label.setText(file_info['description'])
        
        from datetime import datetime
        modified_time = datetime.fromtimestamp(file_info['modified']).strftime('%Y-%m-%d %H:%M:%S')
        self.file_modified_label.setText(modified_time)
        
        accessible_text = "Yes" if file_info['is_accessible'] else "No"
        self.file_accessible_label.setText(accessible_text)
        
        # Update metadata tab
        metadata = self.metadata_manager.get_file_metadata(file_path)
        self.metadata_tags.setText(', '.join(metadata.get('tags', [])))
        self.metadata_description.setPlainText(metadata.get('description', ''))
        self.metadata_category.setCurrentText(metadata.get('category', ''))
        self.metadata_priority.setCurrentText(metadata.get('priority', 'Normal'))
        self.metadata_vector_indexed.setChecked(metadata.get('indexed_for_vector_search', False))
        
        # Update preview tab
        self.load_file_preview(file_path)
        
    def load_file_preview(self, file_path: str):
        """Load file preview content"""
        try:
            file_path_obj = Path(file_path)
            
            if file_path_obj.suffix.lower() in ['.txt', '.md', '.py', '.sql', '.json', '.csv']:
                # Text-based files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Limit preview size
                if len(content) > 5000:
                    content = content[:5000] + "\n\n... (truncated)"
                    
                self.preview_content.setPlainText(content)
            else:
                # Binary or unsupported files
                self.preview_content.setPlainText(f"Preview not available for {file_path_obj.suffix} files")
                
        except Exception as e:
            self.preview_content.setPlainText(f"Error loading preview: {str(e)}")
            
    def save_file_metadata(self):
        """Save metadata for the currently selected file"""
        # Get current selection
        current_file = None
        
        if self.view_mode.currentText() == "List View":
            selection = self.file_tree.selectionModel()
            if selection.hasSelection():
                index = selection.currentIndex()
                if index.isValid():
                    name_item = self.file_model.item(index.row(), 0)
                    current_file = name_item.data(Qt.ItemDataRole.UserRole)
        else:
            current_item = self.file_list.currentItem()
            if current_item:
                current_file = current_item.data(Qt.ItemDataRole.UserRole)
                
        if not current_file:
            QMessageBox.warning(self, "No Selection", "Please select a file to save metadata for.")
            return
            
        # Collect metadata
        tags_text = self.metadata_tags.text().strip()
        tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()] if tags_text else []
        
        metadata = {
            'tags': tags,
            'description': self.metadata_description.toPlainText().strip(),
            'category': self.metadata_category.currentText(),
            'priority': self.metadata_priority.currentText(),
            'indexed_for_vector_search': self.metadata_vector_indexed.isChecked(),
            'custom_fields': {}
        }
        
        # Save metadata
        self.metadata_manager.set_file_metadata(current_file, metadata)
        
        # Emit signal
        self.file_metadata_changed.emit(current_file, metadata)
        
        # Refresh display
        self.apply_filters()
        
        QMessageBox.information(self, "Metadata Saved", "File metadata has been saved successfully.")
            
    def show_context_menu(self, position):
        """Show context menu for file operations"""
        index = self.file_tree.indexAt(position)
        if not index.isValid():
            return
            
        name_item = self.file_model.item(index.row(), 0)
        file_path = name_item.data(Qt.ItemDataRole.UserRole)
        
        menu = QMenu(self)
        
        open_action = QAction("Open", self)
        open_action.triggered.connect(lambda: self.open_file(file_path))
        menu.addAction(open_action)
        
        open_folder_action = QAction("Open Containing Folder", self)
        open_folder_action.triggered.connect(lambda: self.open_containing_folder(file_path))
        menu.addAction(open_folder_action)
        
        menu.addSeparator()
        
        properties_action = QAction("Properties", self)
        properties_action.triggered.connect(lambda: self.show_file_properties(file_path))
        menu.addAction(properties_action)
        
        menu.exec(self.file_tree.mapToGlobal(position))
        
    def open_file(self, file_path: str):
        """Open file with system default application"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS and Linux
                os.system(f'open "{file_path}"' if sys.platform == 'darwin' else f'xdg-open "{file_path}"')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")
            
    def open_containing_folder(self, file_path: str):
        """Open the folder containing the file"""
        folder_path = str(Path(file_path).parent)
        try:
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            elif os.name == 'posix':  # macOS and Linux
                os.system(f'open "{folder_path}"' if sys.platform == 'darwin' else f'xdg-open "{folder_path}"')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
            
    def show_file_properties(self, file_path: str):
        """Show file properties dialog"""
        file_info = None
        for info in self.indexed_files:
            if info['path'] == file_path:
                file_info = info
                break
                
        if file_info:
            from datetime import datetime
            
            properties_text = f"""
File: {file_info['name']}
Path: {file_info['path']}
Type: {file_info['description']}
Size: {file_info['size']} bytes
Modified: {datetime.fromtimestamp(file_info['modified']).strftime('%Y-%m-%d %H:%M:%S')}
Accessible: {'Yes' if file_info['is_accessible'] else 'No'}
Directory: {file_info['directory']}
            """.strip()
            
            QMessageBox.information(self, "File Properties", properties_text)
            
    def refresh(self):
        """Refresh file list"""
        self.start_indexing()
        
    def get_selected_file(self) -> Optional[str]:
        """Get currently selected file path"""
        selection = self.file_tree.selectionModel()
        if selection.hasSelection():
            index = selection.currentIndex()
            if index.isValid():
                name_item = self.file_model.item(index.row(), 0)
                return name_item.data(Qt.ItemDataRole.UserRole)
        return None

class DirectoryMetadataDialog(QDialog):
    """Dialog for collecting metadata when adding a directory"""
    
    def __init__(self, directory_path: str, parent=None):
        super().__init__(parent)
        self.directory_path = directory_path
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the metadata dialog UI"""
        self.setWindowTitle("Directory Metadata")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel(f"<h3>Add Metadata for Directory</h3>")
        layout.addWidget(header_label)
        
        path_label = QLabel(f"<b>Path:</b> {self.directory_path}")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)
        
        # Form layout
        form_layout = QFormLayout()
        
        # Fileset name
        self.fileset_name = QLineEdit()
        self.fileset_name.setText(Path(self.directory_path).name)
        self.fileset_name.setPlaceholderText("e.g., customer_data, sales_reports")
        form_layout.addRow("Dataset Name:", self.fileset_name)
        
        # Description
        self.description = QTextEdit()
        self.description.setMaximumHeight(100)
        self.description.setPlaceholderText("Describe what this dataset contains and its purpose...")
        form_layout.addRow("Description:", self.description)
        
        # Tags
        self.tags = QLineEdit()
        self.tags.setPlaceholderText("customer, analytics, pii, sales (comma-separated)")
        form_layout.addRow("Tags:", self.tags)
        
        # Schema information
        self.schema_info = QTextEdit()
        self.schema_info.setMaximumHeight(80)
        self.schema_info.setPlaceholderText("Describe the data structure, columns, or file formats...")
        form_layout.addRow("Schema/Structure:", self.schema_info)
        
        # Auto-detect file types
        self.auto_detect_btn = QPushButton("Auto-Detect File Types & Schema")
        self.auto_detect_btn.clicked.connect(self.auto_detect_files)
        form_layout.addRow("", self.auto_detect_btn)
        
        # File type summary
        self.file_summary = QLabel("Click 'Auto-Detect' to see file types and suggested schema")
        self.file_summary.setStyleSheet("color: #666; font-style: italic;")
        self.file_summary.setWordWrap(True)
        form_layout.addRow("File Types:", self.file_summary)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        skip_btn = QPushButton("Skip Metadata")
        skip_btn.clicked.connect(self.reject)
        button_layout.addWidget(skip_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Add Directory")
        save_btn.setStyleSheet("""
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
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        layout.addWidget(QFrame())  # Spacer
        layout.addLayout(button_layout)
        
    def auto_detect_files(self):
        """Auto-detect file types and suggest schema in the directory"""
        try:
            directory = Path(self.directory_path)
            file_types = {}
            total_files = 0
            sample_files = {}  # Store sample files for schema detection
            
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    ext = file_path.suffix.lower()
                    if ext:
                        file_types[ext] = file_types.get(ext, 0) + 1
                        # Store first few files of each type for schema analysis
                        if ext not in sample_files:
                            sample_files[ext] = []
                        if len(sample_files[ext]) < 3:  # Keep up to 3 samples per type
                            sample_files[ext].append(file_path)
                    else:
                        file_types['(no extension)'] = file_types.get('(no extension)', 0) + 1
                        
            # Create summary
            if file_types:
                summary_parts = []
                for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                    summary_parts.append(f"{ext}: {count}")
                    
                summary = f"Found {total_files} files: " + ", ".join(summary_parts)
                if len(file_types) > 5:
                    summary += f" and {len(file_types) - 5} more types"
                    
                self.file_summary.setText(summary)
                
                # Auto-suggest tags based on file types
                suggested_tags = []
                if '.csv' in file_types or '.xlsx' in file_types:
                    suggested_tags.append('data')
                    suggested_tags.append('tabular')
                if '.py' in file_types:
                    suggested_tags.append('code')
                    suggested_tags.append('python')
                if '.md' in file_types or '.txt' in file_types:
                    suggested_tags.append('documentation')
                    suggested_tags.append('text')
                if '.json' in file_types:
                    suggested_tags.append('config')
                    suggested_tags.append('structured')
                if '.pdf' in file_types:
                    suggested_tags.append('documents')
                    suggested_tags.append('reports')
                    
                if suggested_tags and not self.tags.text():
                    self.tags.setText(', '.join(suggested_tags))
                    
                # Auto-suggest schema information
                schema_suggestions = self._analyze_schema(sample_files, file_types)
                if schema_suggestions and not self.schema_info.toPlainText():
                    self.schema_info.setPlainText(schema_suggestions)
                    
            else:
                self.file_summary.setText("No files found in directory")
                
        except Exception as e:
            self.file_summary.setText(f"Error scanning directory: {str(e)}")
            
    def _analyze_schema(self, sample_files: Dict[str, list], file_types: Dict[str, int]) -> str:
        """Analyze sample files to suggest schema information"""
        schema_parts = []
        
        try:
            # Analyze CSV files
            if '.csv' in sample_files:
                csv_schema = self._analyze_csv_schema(sample_files['.csv'])
                if csv_schema:
                    schema_parts.append(f"CSV Files ({file_types['.csv']} files):\n{csv_schema}")
                    
            # Analyze JSON files
            if '.json' in sample_files:
                json_schema = self._analyze_json_schema(sample_files['.json'])
                if json_schema:
                    schema_parts.append(f"JSON Files ({file_types['.json']} files):\n{json_schema}")
                    
            # Analyze Python files
            if '.py' in sample_files:
                py_schema = self._analyze_python_schema(sample_files['.py'])
                if py_schema:
                    schema_parts.append(f"Python Files ({file_types['.py']} files):\n{py_schema}")
                    
            # Add general file type summary
            if len(file_types) > 3:
                other_types = [ext for ext in file_types.keys() if ext not in ['.csv', '.json', '.py']]
                if other_types:
                    schema_parts.append(f"Other file types: {', '.join(other_types[:5])}")
                    
        except Exception as e:
            logger.warning(f"Error analyzing schema: {e}")
            
        return '\n\n'.join(schema_parts) if schema_parts else ""
        
    def _analyze_csv_schema(self, csv_files: list) -> str:
        """Analyze CSV files to determine schema"""
        try:
            import csv
            
            for csv_file in csv_files[:2]:  # Check first 2 CSV files
                try:
                    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read first few lines to detect structure
                        sample_lines = [f.readline().strip() for _ in range(5)]
                        sample_lines = [line for line in sample_lines if line]
                        
                        if sample_lines:
                            # Try to detect delimiter
                            sniffer = csv.Sniffer()
                            delimiter = sniffer.sniff(sample_lines[0]).delimiter
                            
                            # Parse header
                            reader = csv.reader([sample_lines[0]], delimiter=delimiter)
                            headers = next(reader)
                            
                            if len(headers) > 1:
                                return f"Columns: {', '.join(headers[:10])}{'...' if len(headers) > 10 else ''} ({len(headers)} total columns)"
                                
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        return "Tabular data files (structure detection failed)"
        
    def _analyze_json_schema(self, json_files: list) -> str:
        """Analyze JSON files to determine schema"""
        try:
            import json
            
            for json_file in json_files[:2]:  # Check first 2 JSON files
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        if isinstance(data, dict):
                            keys = list(data.keys())[:10]
                            return f"Object with keys: {', '.join(keys)}{'...' if len(data) > 10 else ''}"
                        elif isinstance(data, list) and data and isinstance(data[0], dict):
                            keys = list(data[0].keys())[:10]
                            return f"Array of objects with keys: {', '.join(keys)}{'...' if len(data[0]) > 10 else ''}"
                        elif isinstance(data, list):
                            return f"Array with {len(data)} items"
                        else:
                            return f"JSON data of type: {type(data).__name__}"
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        return "Structured JSON data"
        
    def _analyze_python_schema(self, py_files: list) -> str:
        """Analyze Python files to determine content type"""
        try:
            keywords = {
                'class ': 'classes',
                'def ': 'functions', 
                'import ': 'modules',
                'from ': 'imports',
                'if __name__': 'scripts'
            }
            
            found_patterns = set()
            
            for py_file in py_files[:3]:  # Check first 3 Python files
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        for keyword, pattern_type in keywords.items():
                            if keyword in content:
                                found_patterns.add(pattern_type)
                                
                except Exception:
                    continue
                    
            if found_patterns:
                return f"Python code containing: {', '.join(sorted(found_patterns))}"
                
        except Exception:
            pass
            
        return "Python source code"
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get the collected metadata"""
        tags_list = [tag.strip() for tag in self.tags.text().split(',') if tag.strip()]
        
        return {
            'fileset_name': self.fileset_name.text().strip(),
            'description': self.description.toPlainText().strip(),
            'schema_info': self.schema_info.toPlainText().strip(),
            'tags': tags_list,
            'directory_path': self.directory_path
        }
            
    def _detect_csv_schema(self, file_path: Path, content: str) -> str:
        """Detect CSV schema"""
        try:
            import csv
            from io import StringIO
            
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
            
    def _detect_json_schema(self, file_path: Path, content: str) -> str:
        """Detect JSON schema"""
        try:
            import json
            
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
            
    def _detect_python_schema(self, content: str) -> str:
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
            
    def _detect_text_schema(self, content: str) -> str:
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
            
    def _generate_fileset_name(self, file_path: str) -> str:
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
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get the collected metadata"""
        tags_list = [tag.strip() for tag in self.tags.text().split(',') if tag.strip()]
        
        return {
            'fileset_name': self.fileset_name.text().strip(),
            'description': self.description.toPlainText().strip(),
            'schema_info': self.schema_info.toPlainText().strip(),
            'tags': tags_list,
            'directory_path': self.directory_path
        }
