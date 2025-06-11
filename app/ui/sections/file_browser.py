"""
File Browser Widget
Shows files in watched directories with preview and management capabilities
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeView, QLabel, 
    QPushButton, QLineEdit, QComboBox, QMenu, QMessageBox,
    QFileDialog, QFrame, QProgressBar, QDialog, QFormLayout, QTextEdit
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QModelIndex, QThread,
)
from PyQt6.QtGui import QFont, QAction, QStandardItemModel, QStandardItem
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

class FileBrowser(QWidget):
    """File browser widget for showing and managing files"""
    
    file_selected = pyqtSignal(str)  # file_path
    file_double_clicked = pyqtSignal(str)  # file_path
    directory_added = pyqtSignal(str)  # directory_path
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.indexed_files = []
        self.current_filter = ""
        self.indexer_worker = None
        
        self.setup_ui()
        self.setup_connections()
        self.load_watched_directories()
        
    def setup_ui(self):
        """Setup the file browser UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with controls
        header_layout = QVBoxLayout()
        
        # Title and add directory button
        title_layout = QHBoxLayout()
        
        title_label = QLabel("Files")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        add_dir_button = QPushButton("+ Add")
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
        
        # Search and filter
        filter_layout = QHBoxLayout()
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter files...")
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
        self.type_filter.addItems(["All", ".pdf", ".docx", ".xlsx", ".csv", ".json", ".txt"])
        self.type_filter.setStyleSheet("font-size: 9pt;")
        filter_layout.addWidget(self.type_filter)
        
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
        
        # File list
        self.file_model = QStandardItemModel()
        self.file_model.setHorizontalHeaderLabels(['Name', 'Type', 'Size'])
        
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
                padding: 4px;
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
        
        layout.addWidget(self.file_tree)
        
        # Status label
        self.status_label = QLabel("No directories being watched")
        self.status_label.setStyleSheet("font-size: 8pt; color: #666; padding: 4px;")
        layout.addWidget(self.status_label)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.filter_input.textChanged.connect(self.apply_filter)
        self.type_filter.currentTextChanged.connect(self.apply_filter)
        self.file_tree.clicked.connect(self.on_file_clicked)
        self.file_tree.doubleClicked.connect(self.on_file_double_clicked)
        
        # Setup context menu
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.show_context_menu)
        
    def load_watched_directories(self):
        """Load watched directories from config"""
        watched_dirs = self.config_manager.get("file_management.watched_directories", [])
        
        if watched_dirs:
            self.start_indexing()
            self.status_label.setText(f"Watching {len(watched_dirs)} directories")
        else:
            self.status_label.setText("No directories being watched")
            
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
        self.start_indexing()
        # Store metadata for use during vector indexing
        self.pending_metadata = metadata
        
    def trigger_vector_indexing(self):
        """Trigger vector indexing of watched directories"""
        try:
            from utils.vector_search import VectorSearchEngine
            
            vector_engine = VectorSearchEngine(self.config_manager)
            watched_dirs = self.config_manager.get("file_management.watched_directories", [])
            
            for directory in watched_dirs:
                if Path(directory).exists():
                    # Use pending metadata if available
                    metadata = getattr(self, 'pending_metadata', {})
                    
                    vector_engine.index_directory(
                        directory,
                        fileset_name=metadata.get('fileset_name'),
                        fileset_description=metadata.get('description'),
                        tags=metadata.get('tags', [])
                    )
                    
            # Clear pending metadata
            if hasattr(self, 'pending_metadata'):
                delattr(self, 'pending_metadata')
                
        except Exception as e:
            logger.warning(f"Vector indexing failed: {e}")
        
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
        """Add file to the tree model"""
        name_item = QStandardItem(file_info['name'])
        name_item.setData(file_info['path'], Qt.ItemDataRole.UserRole)
        name_item.setToolTip(f"{file_info['path']}\n{file_info['description']}")
        
        # Set icon based on file type
        type_item = QStandardItem(file_info['type'])
        
        # Format file size
        size = file_info['size']
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size // 1024} KB"
        else:
            size_str = f"{size // (1024 * 1024)} MB"
            
        size_item = QStandardItem(size_str)
        size_item.setData(size, Qt.ItemDataRole.UserRole)  # Store actual size for sorting
        
        # Add accessibility indicator
        if not file_info['is_accessible']:
            name_item.setForeground(Qt.GlobalColor.red)
            name_item.setToolTip(f"{file_info['path']}\nAccess denied")
            
        self.file_model.appendRow([name_item, type_item, size_item])
        
    def passes_filter(self, file_info: Dict[str, Any]) -> bool:
        """Check if file passes current filters"""
        # Text filter
        if self.current_filter:
            if self.current_filter.lower() not in file_info['name'].lower():
                return False
                
        # Type filter
        type_filter = self.type_filter.currentText()
        if type_filter != "All" and file_info['type'] != type_filter:
            return False
            
        return True
        
    def apply_filter(self):
        """Apply current filters to file list"""
        self.current_filter = self.filter_input.text()
        
        # Clear and repopulate model
        self.file_model.clear()
        self.file_model.setHorizontalHeaderLabels(['Name', 'Type', 'Size'])
        
        for file_info in self.indexed_files:
            if self.passes_filter(file_info):
                self.add_file_to_model(file_info)
                
        # Update status
        visible_count = self.file_model.rowCount()
        total_count = len(self.indexed_files)
        
        if visible_count != total_count:
            self.status_label.setText(f"Showing {visible_count} of {total_count} files")
        else:
            self.status_label.setText(f"Found {total_count} files in watched directories")
            
    def on_file_clicked(self, index: QModelIndex):
        """Handle file selection"""
        if index.isValid():
            name_item = self.file_model.item(index.row(), 0)
            file_path = name_item.data(Qt.ItemDataRole.UserRole)
            self.file_selected.emit(file_path)
            
    def on_file_double_clicked(self, index: QModelIndex):
        """Handle file double-click"""
        if index.isValid():
            name_item = self.file_model.item(index.row(), 0)
            file_path = name_item.data(Qt.ItemDataRole.UserRole)
            self.file_double_clicked.emit(file_path)
            
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
        
        # Auto-detect file types
        self.auto_detect_btn = QPushButton("Auto-Detect File Types")
        self.auto_detect_btn.clicked.connect(self.auto_detect_files)
        form_layout.addRow("", self.auto_detect_btn)
        
        # File type summary
        self.file_summary = QLabel("Click 'Auto-Detect' to see file types in this directory")
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
        """Auto-detect file types in the directory"""
        try:
            directory = Path(self.directory_path)
            file_types = {}
            total_files = 0
            
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_files += 1
                    ext = file_path.suffix.lower()
                    if ext:
                        file_types[ext] = file_types.get(ext, 0) + 1
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
                if '.py' in file_types:
                    suggested_tags.append('code')
                if '.md' in file_types or '.txt' in file_types:
                    suggested_tags.append('documentation')
                if '.json' in file_types:
                    suggested_tags.append('config')
                    
                if suggested_tags and not self.tags.text():
                    self.tags.setText(', '.join(suggested_tags))
            else:
                self.file_summary.setText("No files found in directory")
                
        except Exception as e:
            self.file_summary.setText(f"Error scanning directory: {str(e)}")
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get the collected metadata"""
        tags_list = [tag.strip() for tag in self.tags.text().split(',') if tag.strip()]
        
        return {
            'fileset_name': self.fileset_name.text().strip(),
            'description': self.description.toPlainText().strip(),
            'tags': tags_list,
            'directory_path': self.directory_path
        }
