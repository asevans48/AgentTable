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
    QListWidget, QListWidgetItem, QSplitter, QGroupBox, QCheckBox, QTabWidget,
    QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QModelIndex, QThread, QFileSystemWatcher, QDir, QFileInfo
)
from PyQt6.QtGui import (
    QFont, QAction, QStandardItemModel, QStandardItem, QIcon, QPixmap, QPainter, QColor,
    QFileSystemModel
)
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
    """File browser widget with proper file system navigation"""
    
    file_selected = pyqtSignal(str)  # file_path
    file_double_clicked = pyqtSignal(str)  # file_path
    directory_added = pyqtSignal(str)  # directory_path
    file_metadata_changed = pyqtSignal(str, dict)  # file_path, metadata
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.metadata_manager = FileMetadataManager(config_manager)
        self.file_watcher = QFileSystemWatcher()
        
        # File system models
        self.dir_model = QFileSystemModel()
        self.file_model = QFileSystemModel()
        
        # Current state
        self.current_directory = None
        self.watched_directories = []
        
        self.setup_ui()
        self.setup_file_system_models()
        self.setup_connections()
        self.load_watched_directories()
        
    def setup_ui(self):
        """Setup the file browser UI like a real file system browser"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with navigation and controls
        header_layout = QVBoxLayout()
        
        # Title and main controls
        title_layout = QHBoxLayout()
        
        title_label = QLabel("File Browser")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Navigation buttons
        self.back_button = QPushButton("◀ Back")
        self.back_button.setEnabled(False)
        self.back_button.setToolTip("Go back")
        title_layout.addWidget(self.back_button)
        
        self.forward_button = QPushButton("▶ Forward")
        self.forward_button.setEnabled(False)
        self.forward_button.setToolTip("Go forward")
        title_layout.addWidget(self.forward_button)
        
        self.up_button = QPushButton("⬆ Up")
        self.up_button.setEnabled(False)
        self.up_button.setToolTip("Go to parent directory")
        title_layout.addWidget(self.up_button)
        
        self.home_button = QPushButton("🏠 Home")
        self.home_button.setToolTip("Go to home directory")
        title_layout.addWidget(self.home_button)
        
        # Add watched directory button
        add_dir_button = QPushButton("+ Add Directory")
        add_dir_button.setToolTip("Add directory to watch list")
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
        
        # Address bar
        address_layout = QHBoxLayout()
        address_layout.addWidget(QLabel("Location:"))
        
        self.address_bar = QLineEdit()
        self.address_bar.setPlaceholderText("Enter path or select from tree...")
        self.address_bar.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
            }
        """)
        address_layout.addWidget(self.address_bar)
        
        self.go_button = QPushButton("Go")
        self.go_button.setToolTip("Navigate to address")
        address_layout.addWidget(self.go_button)
        
        header_layout.addLayout(address_layout)
        
        # Search and filter bar
        filter_layout = QHBoxLayout()
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search files and folders...")
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
        self.type_filter.addItems(["All Files", "Documents", "Images", "Code", "Data", "Archives"])
        self.type_filter.setStyleSheet("font-size: 9pt;")
        filter_layout.addWidget(self.type_filter)
        
        self.show_hidden = QCheckBox("Show Hidden")
        self.show_hidden.setToolTip("Show hidden files and folders")
        filter_layout.addWidget(self.show_hidden)
        
        header_layout.addLayout(filter_layout)
        
        layout.addLayout(header_layout)
        
        # Main browser area with splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Directory tree
        left_panel = QFrame()
        left_panel.setMaximumWidth(300)
        left_panel.setMinimumWidth(200)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Watched directories section
        watched_label = QLabel("Watched Directories")
        watched_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        watched_label.setStyleSheet("padding: 4px; background-color: #f0f0f0; border-radius: 3px;")
        left_layout.addWidget(watched_label)
        
        self.watched_tree = QTreeView()
        self.watched_tree.setMaximumHeight(120)
        self.watched_tree.setHeaderHidden(True)
        self.watched_tree.setStyleSheet("""
            QTreeView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            QTreeView::item {
                padding: 4px;
            }
            QTreeView::item:hover {
                background-color: #e9ecef;
            }
            QTreeView::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        left_layout.addWidget(self.watched_tree)
        
        # Directory tree section
        tree_label = QLabel("Directory Tree")
        tree_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        tree_label.setStyleSheet("padding: 4px; background-color: #f0f0f0; border-radius: 3px; margin-top: 8px;")
        left_layout.addWidget(tree_label)
        
        self.directory_tree = QTreeView()
        self.directory_tree.setHeaderHidden(True)
        self.directory_tree.setStyleSheet("""
            QTreeView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QTreeView::item {
                padding: 3px;
            }
            QTreeView::item:hover {
                background-color: #f5f5f5;
            }
            QTreeView::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        left_layout.addWidget(self.directory_tree)
        
        main_splitter.addWidget(left_panel)
        
        # Center panel: File list
        center_panel = QFrame()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # View controls
        view_controls = QHBoxLayout()
        
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Details", "List", "Icons"])
        self.view_mode.setCurrentText("Details")
        view_controls.addWidget(QLabel("View:"))
        view_controls.addWidget(self.view_mode)
        
        view_controls.addStretch()
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name", "Size", "Type", "Date Modified"])
        view_controls.addWidget(QLabel("Sort by:"))
        view_controls.addWidget(self.sort_combo)
        
        self.sort_order = QComboBox()
        self.sort_order.addItems(["Ascending", "Descending"])
        view_controls.addWidget(self.sort_order)
        
        center_layout.addLayout(view_controls)
        
        # File list view
        self.file_list = QTreeView()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setSortingEnabled(True)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.setStyleSheet("""
            QTreeView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                selection-background-color: #e3f2fd;
                gridline-color: #f0f0f0;
            }
            QTreeView::item {
                padding: 4px;
                border-bottom: 1px solid #f8f9fa;
            }
            QTreeView::item:hover {
                background-color: #f5f5f5;
            }
            QTreeView::item:selected {
                background-color: #e3f2fd;
                color: black;
            }
        """)
        center_layout.addWidget(self.file_list)
        
        main_splitter.addWidget(center_panel)
        
        # Right panel: File details and metadata
        self.details_panel = self.create_details_panel()
        main_splitter.addWidget(self.details_panel)
        
        # Set splitter proportions (tree, files, details)
        main_splitter.setSizes([250, 500, 300])
        
        layout.addWidget(main_splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 8pt; color: #666; padding: 4px;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("font-size: 8pt; color: #666; padding: 4px;")
        status_layout.addWidget(self.selection_label)
        
        layout.addLayout(status_layout)
        
    def go_back(self):
        """Go back in navigation history"""
        # TODO: Implement navigation history
        pass
        
    def go_forward(self):
        """Go forward in navigation history"""
        # TODO: Implement navigation history
        pass
        
    def go_up(self):
        """Go to parent directory"""
        if self.current_directory:
            parent = str(Path(self.current_directory).parent)
            if parent != self.current_directory:
                self.navigate_to_directory(parent)
                
    def go_home(self):
        """Go to home directory"""
        self.navigate_to_directory(str(Path.home()))
        
    def navigate_to_address(self):
        """Navigate to the address in the address bar"""
        path = self.address_bar.text().strip()
        if path and Path(path).exists():
            self.navigate_to_directory(path)
        else:
            QMessageBox.warning(self, "Invalid Path", f"The path '{path}' does not exist.")
            
    def toggle_hidden_files(self, show_hidden):
        """Toggle display of hidden files"""
        if show_hidden:
            self.file_model.setFilter(QDir.Filter.AllEntries)
            self.dir_model.setFilter(QDir.Filter.Dirs | QDir.Filter.Hidden)
        else:
            self.file_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
            self.dir_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
            
    def change_view_mode(self, mode):
        """Change file list view mode"""
        # TODO: Implement different view modes (Details, List, Icons)
        pass
        
    def change_sort(self):
        """Change file list sorting"""
        sort_column = self.sort_combo.currentIndex()
        sort_order = Qt.SortOrder.AscendingOrder if self.sort_order.currentText() == "Ascending" else Qt.SortOrder.DescendingOrder
        self.file_list.sortByColumn(sort_column, sort_order)
        
    def on_directory_tree_clicked(self, index):
        """Handle directory tree clicks"""
        if index.isValid():
            path = self.dir_model.filePath(index)
            self.navigate_to_directory(path)
            
    def on_watched_directory_clicked(self, index):
        """Handle watched directory clicks"""
        if index.isValid():
            item = self.watched_model.itemFromIndex(index)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self.navigate_to_directory(path)
                
    def on_file_clicked(self, index):
        """Handle file clicks"""
        if index.isValid():
            file_info = self.file_model.fileInfo(index)
            if file_info.isDir():
                # Double-click to enter directory, single click to select
                return
            else:
                # Show file details
                self.show_file_details(file_info.absoluteFilePath())
                self.file_selected.emit(file_info.absoluteFilePath())
                
    def on_file_double_clicked(self, index):
        """Handle file double-clicks"""
        if index.isValid():
            file_info = self.file_model.fileInfo(index)
            if file_info.isDir():
                # Navigate into directory
                self.navigate_to_directory(file_info.absoluteFilePath())
            else:
                # Open file
                self.open_file(file_info.absoluteFilePath())
                self.file_double_clicked.emit(file_info.absoluteFilePath())
                
    def on_selection_changed(self, selected, deselected):
        """Handle selection changes"""
        indexes = self.file_list.selectionModel().selectedIndexes()
        if indexes:
            # Show count of selected items
            selected_count = len(set(index.row() for index in indexes))
            if selected_count == 1:
                file_info = self.file_model.fileInfo(indexes[0])
                self.selection_label.setText(f"Selected: {file_info.fileName()}")
            else:
                self.selection_label.setText(f"Selected: {selected_count} items")
        else:
            self.selection_label.setText("")
            
    def update_status(self):
        """Update status bar with current directory info"""
        if self.current_directory:
            dir_path = Path(self.current_directory)
            try:
                # Count items in current directory
                items = list(dir_path.iterdir())
                file_count = sum(1 for item in items if item.is_file())
                dir_count = sum(1 for item in items if item.is_dir())
                
                status_parts = []
                if dir_count > 0:
                    status_parts.append(f"{dir_count} folders")
                if file_count > 0:
                    status_parts.append(f"{file_count} files")
                    
                if status_parts:
                    self.status_label.setText(f"{dir_path.name}: {', '.join(status_parts)}")
                else:
                    self.status_label.setText(f"{dir_path.name}: Empty")
                    
            except PermissionError:
                self.status_label.setText(f"{dir_path.name}: Access denied")
            except Exception as e:
                self.status_label.setText(f"{dir_path.name}: Error - {str(e)}")
        else:
            self.status_label.setText("No directory selected")
            
    def apply_filters(self):
        """Apply current filters to file list"""
        filter_text = self.filter_input.text().strip()
        
        if filter_text:
            # Set name filter
            self.file_model.setNameFilters([f"*{filter_text}*"])
            self.file_model.setNameFilterDisables(False)
        else:
            # Clear name filter
            self.file_model.setNameFilters([])
            
        # TODO: Implement type filtering based on type_filter combo
        
    def setup_file_system_models(self):
        """Setup file system models for directory tree and file list"""
        # Directory model - shows only directories
        self.dir_model.setRootPath("")
        self.dir_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        
        # File model - shows files and directories
        self.file_model.setRootPath("")
        self.file_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
        
        # Set models to views
        self.directory_tree.setModel(self.dir_model)
        self.file_list.setModel(self.file_model)
        
        # Hide extra columns in directory tree (only show name)
        for i in range(1, self.dir_model.columnCount()):
            self.directory_tree.hideColumn(i)
            
        # Configure file list columns
        header = self.file_list.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Size
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Type
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Date
        
        # Set initial directory to user's home
        home_path = str(Path.home())
        self.navigate_to_directory(home_path)
        
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
        
        
    def navigate_to_directory(self, directory_path):
        """Navigate to a specific directory"""
        if not directory_path or not Path(directory_path).exists():
            return
            
        self.current_directory = directory_path
        
        # Update address bar
        self.address_bar.setText(directory_path)
        
        # Set root path for file list
        index = self.file_model.setRootPath(directory_path)
        self.file_list.setRootIndex(index)
        
        # Expand directory tree to show current location
        dir_index = self.dir_model.index(directory_path)
        if dir_index.isValid():
            self.directory_tree.setCurrentIndex(dir_index)
            self.directory_tree.scrollTo(dir_index)
            
        # Update navigation buttons
        self.up_button.setEnabled(Path(directory_path).parent != Path(directory_path))
        
        # Apply current filters
        self.apply_filters()
        
        # Update status
        self.update_status()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        # Navigation buttons
        self.back_button.clicked.connect(self.go_back)
        self.forward_button.clicked.connect(self.go_forward)
        self.up_button.clicked.connect(self.go_up)
        self.home_button.clicked.connect(self.go_home)
        self.go_button.clicked.connect(self.navigate_to_address)
        
        # Address bar
        self.address_bar.returnPressed.connect(self.navigate_to_address)
        
        # Filters and search
        self.filter_input.textChanged.connect(self.apply_filters)
        self.type_filter.currentTextChanged.connect(self.apply_filters)
        self.show_hidden.toggled.connect(self.toggle_hidden_files)
        
        # View controls
        self.view_mode.currentTextChanged.connect(self.change_view_mode)
        self.sort_combo.currentTextChanged.connect(self.change_sort)
        self.sort_order.currentTextChanged.connect(self.change_sort)
        
        # Tree and file list connections
        self.directory_tree.clicked.connect(self.on_directory_tree_clicked)
        self.file_list.clicked.connect(self.on_file_clicked)
        self.file_list.doubleClicked.connect(self.on_file_double_clicked)
        
        # Connect selection model if it exists
        if self.file_list.selectionModel():
            self.file_list.selectionModel().selectionChanged.connect(self.on_selection_changed)
        
        # Watched directories
        self.watched_tree.clicked.connect(self.on_watched_directory_clicked)
        
        # Context menus
        self.file_list.customContextMenuRequested.connect(self.show_context_menu)
        
        # File watcher
        self.file_watcher.directoryChanged.connect(self.on_directory_changed)
        
    def load_watched_directories(self):
        """Load watched directories from config"""
        self.watched_directories = self.config_manager.get("file_management.watched_directories", [])
        
        # Create model for watched directories
        self.watched_model = QStandardItemModel()
        self.watched_tree.setModel(self.watched_model)
        
        # Populate watched directories
        for directory in self.watched_directories:
            if Path(directory).exists():
                dir_item = QStandardItem(f"📂 {Path(directory).name}")
                dir_item.setData(directory, Qt.ItemDataRole.UserRole)
                dir_item.setToolTip(directory)
                self.watched_model.appendRow(dir_item)
                
                # Add to file watcher
                self.file_watcher.addPath(directory)
        
        # Update status
        if self.watched_directories:
            self.status_label.setText(f"Watching {len(self.watched_directories)} directories")
        else:
            self.status_label.setText("No directories being watched - add some to get started")
            
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
            
    def refresh(self):
        """Refresh the current directory view"""
        if self.current_directory:
            self.navigate_to_directory(self.current_directory)
        self.update_status()
        
    def get_selected_file(self) -> Optional[str]:
        """Get currently selected file path"""
        indexes = self.file_list.selectionModel().selectedIndexes()
        if indexes:
            file_info = self.file_model.fileInfo(indexes[0])
            return file_info.absoluteFilePath()
        return None
        
        
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
        if not file_path or not Path(file_path).exists():
            self.details_tabs.setVisible(False)
            self.no_selection_label.setVisible(True)
            return
            
        file_info = QFileInfo(file_path)
        
        # Show details tabs
        self.details_tabs.setVisible(True)
        self.no_selection_label.setVisible(False)
        
        # Update properties tab
        self.file_name_label.setText(file_info.fileName())
        self.file_path_label.setText(file_info.absoluteFilePath())
        self.file_size_label.setText(self.format_file_size(file_info.size()))
        
        # File type description
        if file_info.isDir():
            type_desc = "Folder"
        else:
            suffix = file_info.suffix().lower()
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
            type_desc = type_descriptions.get(suffix, f'{suffix.upper()} File' if suffix else 'File')
            
        self.file_type_label.setText(type_desc)
        
        # Modified time
        modified_time = file_info.lastModified().toString('yyyy-MM-dd hh:mm:ss')
        self.file_modified_label.setText(modified_time)
        
        # Accessibility
        accessible_text = "Yes" if file_info.isReadable() else "No"
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
        indexes = self.file_list.selectionModel().selectedIndexes()
        if not indexes:
            QMessageBox.warning(self, "No Selection", "Please select a file to save metadata for.")
            return
            
        # Get the file path from the first selected item
        file_info = self.file_model.fileInfo(indexes[0])
        current_file = file_info.absoluteFilePath()
        
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
        
        QMessageBox.information(self, "Metadata Saved", "File metadata has been saved successfully.")
            
    def show_context_menu(self, position):
        """Show context menu for file operations"""
        index = self.file_list.indexAt(position)
        if not index.isValid():
            return
            
        file_info = self.file_model.fileInfo(index)
        file_path = file_info.absoluteFilePath()
        
        menu = QMenu(self)
        
        if file_info.isDir():
            open_action = QAction("Open Folder", self)
            open_action.triggered.connect(lambda: self.navigate_to_directory(file_path))
            menu.addAction(open_action)
        else:
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
        
        menu.exec(self.file_list.mapToGlobal(position))
        
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
        file_info = QFileInfo(file_path)
        
        if file_info.exists():
            properties_text = f"""
File: {file_info.fileName()}
Path: {file_info.absoluteFilePath()}
Type: {'Folder' if file_info.isDir() else 'File'}
Size: {self.format_file_size(file_info.size())}
Modified: {file_info.lastModified().toString('yyyy-MM-dd hh:mm:ss')}
Created: {file_info.birthTime().toString('yyyy-MM-dd hh:mm:ss')}
Readable: {'Yes' if file_info.isReadable() else 'No'}
Writable: {'Yes' if file_info.isWritable() else 'No'}
Directory: {file_info.absolutePath()}
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
