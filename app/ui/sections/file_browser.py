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
        
        # Current state
        self.watched_directories = []
        self.indexed_files = []
        self.current_filter = ""
        self.indexer_worker = None
        self.current_directory = None
        
        self.setup_ui()
        self.setup_connections()
        self.load_watched_directories()
        
    def setup_ui(self):
        """Setup the compact file browser UI for sidebar use"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Compact header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        # Title and add button
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("Files")
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Compact add button
        add_dir_button = QPushButton("+")
        add_dir_button.setToolTip("Add directory to watch")
        add_dir_button.setFixedSize(24, 24)
        add_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        add_dir_button.clicked.connect(self.add_directory)
        title_layout.addWidget(add_dir_button)
        
        header_layout.addLayout(title_layout)
        
        # Compact search bar
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search files...")
        self.filter_input.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 9pt;
            }
        """)
        header_layout.addWidget(self.filter_input)
        
        # Compact filter row
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", ".pdf", ".csv", ".json", ".txt", ".py"])
        self.type_filter.setStyleSheet("font-size: 8pt; padding: 2px;")
        self.type_filter.setMaximumHeight(24)
        filter_layout.addWidget(self.type_filter)
        
        self.metadata_filter = QComboBox()
        self.metadata_filter.addItems(["All", "Tagged", "Indexed"])
        self.metadata_filter.setStyleSheet("font-size: 8pt; padding: 2px;")
        self.metadata_filter.setMaximumHeight(24)
        filter_layout.addWidget(self.metadata_filter)
        
        header_layout.addLayout(filter_layout)
        layout.addLayout(header_layout)
        
        # Watched directories section
        watched_header = QLabel("Directories")
        watched_header.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        watched_header.setStyleSheet("color: #495057; padding: 2px 0px;")
        layout.addWidget(watched_header)
        
        self.watched_tree = QTreeView()
        self.watched_tree.setHeaderHidden(True)
        self.watched_tree.setMaximumHeight(100)
        self.watched_tree.setStyleSheet("""
            QTreeView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
                font-size: 9pt;
            }
            QTreeView::item {
                padding: 6px 8px;
                border: none;
            }
            QTreeView::item:hover {
                background-color: #e9ecef;
            }
            QTreeView::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        layout.addWidget(self.watched_tree)
        
        # Files section
        files_header_layout = QHBoxLayout()
        files_header_layout.setContentsMargins(0, 8, 0, 0)
        
        files_header = QLabel("Files")
        files_header.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        files_header.setStyleSheet("color: #495057; padding: 2px 0px;")
        files_header_layout.addWidget(files_header)
        
        files_header_layout.addStretch()
        
        # File count label
        self.file_count_label = QLabel("0")
        self.file_count_label.setStyleSheet("color: #6c757d; font-size: 8pt; padding: 2px 4px; background-color: #e9ecef; border-radius: 8px;")
        files_header_layout.addWidget(self.file_count_label)
        
        layout.addLayout(files_header_layout)
        
        # Compact file list
        self.file_list = QTreeView()
        self.file_list.setHeaderHidden(True)
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setRootIsDecorated(False)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.setStyleSheet("""
            QTreeView {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-size: 9pt;
            }
            QTreeView::item {
                padding: 4px 6px;
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
        layout.addWidget(self.file_list)
        
        # Compact status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 8pt; color: #6c757d; padding: 4px 0px;")
        layout.addWidget(self.status_label)
        
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
        """Handle watched directory clicks - show files from that directory"""
        if index.isValid():
            item = self.watched_model.itemFromIndex(index)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self.current_directory = path
                self.apply_filters()
                
                # Update status to show current directory
                dir_name = Path(path).name
                self.status_label.setText(f"📂 {dir_name}")
                
    def on_file_clicked(self, index):
        """Handle file clicks"""
        if index.isValid():
            name_item = self.file_model.item(index.row(), 0)
            file_path = name_item.data(Qt.ItemDataRole.UserRole)
            self.show_file_details(file_path)
            self.file_selected.emit(file_path)
                
    def on_file_double_clicked(self, index):
        """Handle file double-clicks"""
        if index.isValid():
            name_item = self.file_model.item(index.row(), 0)
            file_path = name_item.data(Qt.ItemDataRole.UserRole)
            self.open_file(file_path)
            self.file_double_clicked.emit(file_path)
                
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
        self.current_filter = self.filter_input.text()
        
        # Clear and repopulate models
        self.file_model.clear()
        self.file_model.setHorizontalHeaderLabels(['Files'])
        
        filtered_files = []
        for file_info in self.indexed_files:
            if self.passes_filter(file_info):
                self.add_file_to_model(file_info)
                filtered_files.append(file_info)
                
        # Update file count
        visible_count = len(filtered_files)
        self.file_count_label.setText(str(visible_count))
        
        # Update status
        if self.current_directory:
            dir_name = Path(self.current_directory).name
            self.status_label.setText(f"📂 {dir_name} ({visible_count} files)")
        else:
            self.status_label.setText(f"All directories ({visible_count} files)")
        
    def setup_file_models(self):
        """Setup compact file list model"""
        # Create standard item model for file list - single column for compact view
        self.file_model = QStandardItemModel()
        self.file_model.setHorizontalHeaderLabels(['Files'])
        self.file_list.setModel(self.file_model)
        
    def create_details_panel(self):
        """Create a minimal details panel for the compact sidebar"""
        details_widget = QWidget()
        details_widget.setVisible(False)  # Hidden by default in compact mode
        
        layout = QVBoxLayout(details_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Minimal file info
        self.selected_file_label = QLabel("No file selected")
        self.selected_file_label.setStyleSheet("font-size: 8pt; color: #6c757d; padding: 4px;")
        self.selected_file_label.setWordWrap(True)
        layout.addWidget(self.selected_file_label)
        
        return details_widget
        
        
        
    def refresh(self):
        """Refresh file list"""
        self.start_indexing()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        # Filters and search
        self.filter_input.textChanged.connect(self.apply_filters)
        self.type_filter.currentTextChanged.connect(self.apply_filters)
        self.metadata_filter.currentTextChanged.connect(self.apply_filters)
        
        # File list connections
        self.file_list.clicked.connect(self.on_file_clicked)
        self.file_list.doubleClicked.connect(self.on_file_double_clicked)
        
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
        
        self.setup_file_models()
        self.start_indexing()
        
    def passes_filter(self, file_info: Dict[str, Any]) -> bool:
        """Check if file passes current filters"""
        # Directory filter - show files in the selected directory
        if self.current_directory:
            file_dir = str(Path(file_info['path']).parent)
            # Check if file is directly in the selected directory (not subdirectories)
            if file_dir != self.current_directory:
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
        
    def add_file_to_model(self, file_info: Dict[str, Any]):
        """Add file to the compact file list model"""
        # Get file metadata
        metadata = self.metadata_manager.get_file_metadata(file_info['path'])
        
        # Create file icon based on type
        file_icon = self.get_file_icon(file_info['type'])
        
        # Create compact display text with file name and size
        display_text = file_info['name']
        size_text = self.format_file_size(file_info['size'])
        
        # Add tags indicator if present
        if metadata.get('tags'):
            display_text += f" 🏷️"
        
        # Add to single column view
        name_item = QStandardItem(display_text)
        name_item.setData(file_info['path'], Qt.ItemDataRole.UserRole)
        name_item.setIcon(file_icon)
        
        # Enhanced tooltip with all metadata
        tooltip_parts = [
            f"📁 {file_info['name']}",
            f"📂 {Path(file_info['path']).parent.name}",
            f"📏 {self.format_file_size(file_info['size'])}",
            f"🗂️ {file_info['description']}"
        ]
        if metadata.get('description'):
            tooltip_parts.append(f"📝 {metadata['description']}")
        if metadata.get('tags'):
            tooltip_parts.append(f"🏷️ {', '.join(metadata['tags'])}")
        name_item.setToolTip('\n'.join(tooltip_parts))
        
        # Add accessibility indicator
        if not file_info['is_accessible']:
            name_item.setForeground(Qt.GlobalColor.red)
            
        self.file_model.appendRow([name_item])
        
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
        self.file_model.setHorizontalHeaderLabels(['Name', 'Type', 'Size', 'Tags'])
        self.indexed_files.clear()
        
        # Start new indexer
        self.indexer_worker = FileIndexWorker(watched_dirs, supported_formats)
        self.indexer_worker.file_indexed.connect(self.on_file_indexed)
        self.indexer_worker.indexing_complete.connect(self.on_indexing_complete)
        self.indexer_worker.start()
        
    def on_file_indexed(self, file_info: Dict[str, Any]):
        """Handle newly indexed file"""
        self.indexed_files.append(file_info)
        
        # Add to model if it passes current filter
        if self.passes_filter(file_info):
            self.add_file_to_model(file_info)
            
    def on_indexing_complete(self, total_files: int):
        """Handle indexing completion"""
        self.status_label.setText(f"Ready ({total_files} files)")
        self.file_count_label.setText(str(total_files))
        
        # Sort the tree
        self.file_list.sortByColumn(0, Qt.SortOrder.AscendingOrder)
            
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
                
                # Show message about vector indexing
                QMessageBox.information(
                    self,
                    "Directory Added",
                    f"Directory '{metadata.get('fileset_name', 'Unknown')}' has been added to the file browser.\n\n"
                    "To enable vector search for this directory, go to the Search section and click 'Rebuild Index'."
                )
            else:
                # User cancelled, add without metadata
                self.config_manager.add_watched_directory(directory)
                self.directory_added.emit(directory)
            
            # Reload watched directories to show the new one
            self.load_watched_directories()
            
    def refresh(self):
        """Refresh the current directory view"""
        if self.current_directory:
            self.navigate_to_directory(self.current_directory)
        self.update_status()
        
    def get_selected_file(self) -> Optional[str]:
        """Get currently selected file path"""
        selection = self.file_list.selectionModel()
        if selection.hasSelection():
            index = selection.currentIndex()
            if index.isValid():
                name_item = self.file_model.item(index.row(), 0)
                return name_item.data(Qt.ItemDataRole.UserRole)
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
        # Refresh current view if we're viewing the changed directory
        if self.current_directory and Path(path) == Path(self.current_directory):
            self.refresh()
                
    def show_file_details(self, file_path: str):
        """Show minimal file details in the compact sidebar"""
        if not file_path:
            self.selected_file_label.setText("No file selected")
            return
            
        # Find file info
        file_info = None
        for info in self.indexed_files:
            if info['path'] == file_path:
                file_info = info
                break
                
        if not file_info:
            return
            
        # Show compact file info
        file_name = file_info['name']
        file_size = self.format_file_size(file_info['size'])
        
        # Get metadata for tags
        metadata = self.metadata_manager.get_file_metadata(file_path)
        tags = metadata.get('tags', [])
        
        display_text = f"📁 {file_name}\n📏 {file_size}"
        if tags:
            display_text += f"\n🏷️ {', '.join(tags[:2])}"
            if len(tags) > 2:
                display_text += f" +{len(tags)-2}"
                
        self.selected_file_label.setText(display_text)
        
            
            
    def show_context_menu(self, position):
        """Show context menu for file operations"""
        index = self.file_list.indexAt(position)
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
        
