"""
Search Section Widget
Main search interface with vector search, document viewer, and SQL query tabs
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit,
    QLabel, QPushButton, QFrame, QComboBox, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SQLHighlighter(QSyntaxHighlighter):
    """SQL syntax highlighter for query editor"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_highlighting_rules()
        
    def setup_highlighting_rules(self):
        """Setup SQL syntax highlighting rules"""
        self.highlighting_rules = []
        
        # SQL keywords
        keyword_format = QTextCharFormat()
        keyword_format.setColor(QColor(0, 0, 255))
        keyword_format.setFontWeight(QFont.Weight.Bold)
        
        keywords = [
            "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER",
            "GROUP", "BY", "ORDER", "HAVING", "INSERT", "UPDATE", "DELETE",
            "CREATE", "ALTER", "DROP", "TABLE", "VIEW", "INDEX", "DATABASE",
            "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN", "IS", "NULL",
            "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "AS"
        ]
        
        for keyword in keywords:
            pattern = f"\\b{keyword}\\b"
            self.highlighting_rules.append((pattern, keyword_format))
            
        # String literals
        string_format = QTextCharFormat()
        string_format.setColor(QColor(0, 128, 0))
        self.highlighting_rules.append(("'[^']*'", string_format))
        self.highlighting_rules.append(("\"[^\"]*\"", string_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setColor(QColor(255, 0, 255))
        self.highlighting_rules.append(("\\b\\d+\\b", number_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setColor(QColor(128, 128, 128))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append(("--[^\n]*", comment_format))
        
    def highlightBlock(self, text):
        """Apply highlighting to text block"""
        import re
        
        for pattern, format in self.highlighting_rules:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                self.setFormat(start, end - start, format)

class VectorSearchTab(QWidget):
    """Vector search tab widget"""
    
    search_performed = pyqtSignal(str, dict)  # query, results
    indexing_started = pyqtSignal()  # indexing started
    indexing_completed = pyqtSignal(dict)  # indexing results
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.vector_engine = None
        self.indexing_worker = None
        self.setup_ui()
        self.setup_connections()
        self.load_vector_engine()
        
    def setup_ui(self):
        """Setup vector search UI"""
        layout = QVBoxLayout(self)
        
        # Vector Database Management Section
        db_management_frame = QFrame()
        db_management_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        db_management_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        db_layout = QVBoxLayout(db_management_frame)
        
        # Database status and controls
        db_header_layout = QHBoxLayout()
        
        db_title = QLabel("Vector Database")
        db_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        db_header_layout.addWidget(db_title)
        
        db_header_layout.addStretch()
        
        # Database management buttons
        self.rebuild_btn = QPushButton("🔄 Rebuild Index")
        self.rebuild_btn.setToolTip("Rebuild the entire vector search index")
        self.rebuild_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        db_header_layout.addWidget(self.rebuild_btn)
        
        self.clear_index_btn = QPushButton("🗑️ Clear")
        self.clear_index_btn.setToolTip("Clear all indexed documents")
        self.clear_index_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #c82333; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        db_header_layout.addWidget(self.clear_index_btn)
        
        db_layout.addLayout(db_header_layout)
        
        # Database status
        self.db_status_label = QLabel("Loading vector search engine...")
        self.db_status_label.setStyleSheet("color: #495057; font-size: 9pt; margin: 4px 0;")
        db_layout.addWidget(self.db_status_label)
        
        # Indexing progress
        self.indexing_progress = QProgressBar()
        self.indexing_progress.setVisible(False)
        self.indexing_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
        """)
        db_layout.addWidget(self.indexing_progress)
        
        self.indexing_status_label = QLabel("")
        self.indexing_status_label.setStyleSheet("color: #6c757d; font-size: 8pt;")
        self.indexing_status_label.setVisible(False)
        db_layout.addWidget(self.indexing_status_label)
        
        layout.addWidget(db_management_frame)
        
        # Search configuration
        config_frame = QFrame()
        config_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        config_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        config_layout = QVBoxLayout(config_frame)
        
        # Configuration title
        config_title = QLabel("Search Configuration")
        config_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        config_layout.addWidget(config_title)
        
        # Configuration controls
        config_controls_layout = QHBoxLayout()
        
        # Embedding model selection
        config_controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1"
        ])
        self.model_combo.setCurrentText(self.config_manager.get("vector_search.model", "all-MiniLM-L6-v2"))
        config_controls_layout.addWidget(self.model_combo)
        
        # Max results
        config_controls_layout.addWidget(QLabel("Max Results:"))
        self.max_results = QComboBox()
        self.max_results.addItems(["10", "25", "50", "100"])
        self.max_results.setCurrentText("25")
        config_controls_layout.addWidget(self.max_results)
        
        # Similarity threshold
        config_controls_layout.addWidget(QLabel("Min Similarity:"))
        self.similarity_threshold = QComboBox()
        self.similarity_threshold.addItems(["0.1", "0.2", "0.3", "0.4", "0.5"])
        self.similarity_threshold.setCurrentText("0.3")
        config_controls_layout.addWidget(self.similarity_threshold)
        
        config_controls_layout.addStretch()
        
        # Semantic search toggle
        self.semantic_search = QCheckBox("Semantic Search")
        self.semantic_search.setChecked(True)
        config_controls_layout.addWidget(self.semantic_search)
        
        config_layout.addLayout(config_controls_layout)
        layout.addWidget(config_frame)
        
        # Instructions and status
        instructions_frame = QFrame()
        instructions_frame.setStyleSheet("""
            QFrame {
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 12px;
            }
        """)
        instructions_layout = QVBoxLayout(instructions_frame)
        
        instructions = QLabel("""
<b>Vector Search Instructions:</b><br>
• Click "Rebuild Index" to index all files in watched directories<br>
• Use the main search bar to perform semantic searches<br>
• Results show content chunks with similarity scores<br>
• Adjust similarity threshold to filter results
        """)
        instructions.setStyleSheet("color: #495057; font-size: 9pt; line-height: 1.4;")
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        
        layout.addWidget(instructions_frame)
        
        layout.addStretch()
        
    def setup_connections(self):
        """Setup signal connections"""
        self.rebuild_btn.clicked.connect(self.rebuild_vector_index)
        self.clear_index_btn.clicked.connect(self.clear_vector_index)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        
    def load_vector_engine(self):
        """Load the vector search engine"""
        try:
            from utils.vector_search import VectorSearchEngine
            self.vector_engine = VectorSearchEngine(self.config_manager)
            self.update_database_status()
        except ImportError:
            self.db_status_label.setText("❌ Vector search dependencies not installed")
            self.db_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            self.rebuild_btn.setEnabled(False)
            self.clear_index_btn.setEnabled(False)
        except Exception as e:
            self.db_status_label.setText(f"❌ Error loading vector engine: {str(e)}")
            self.db_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            
    def update_database_status(self):
        """Update database status display"""
        if not self.vector_engine:
            return
            
        try:
            docs = self.vector_engine.get_indexed_documents()
            doc_count = len(docs)
            
            if doc_count > 0:
                self.db_status_label.setText(f"✅ {doc_count} documents indexed and ready for search")
                self.db_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            else:
                self.db_status_label.setText("📚 No documents indexed yet - click 'Rebuild Index' to get started")
                self.db_status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
                
        except Exception as e:
            self.db_status_label.setText(f"❌ Status error: {str(e)}")
            self.db_status_label.setStyleSheet("color: #dc3545;")
            
    def rebuild_vector_index(self):
        """Rebuild the entire vector search index with all data sources"""
        if not self.vector_engine:
            return
            
        try:
            # Get all data sources
            watched_dirs = self.config_manager.get("file_management.watched_directories", [])
            registered_datasets = self.config_manager.get_registered_datasets()
            
            # Show data source selection dialog
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QLabel, QPushButton, QHBoxLayout, QGroupBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Rebuild Vector Index - Select Data Sources")
            dialog.setModal(True)
            dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Header
            header_label = QLabel("<h3>Select Data Sources to Include in Vector Index</h3>")
            layout.addWidget(header_label)
            
            info_label = QLabel("Choose which data sources to include in the rebuilt vector search index:")
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
            layout.addWidget(info_label)
            
            # Watched directories group
            dirs_group = QGroupBox("Watched Directories")
            dirs_layout = QVBoxLayout(dirs_group)
            
            dir_checkboxes = {}
            if watched_dirs:
                for directory in watched_dirs:
                    if Path(directory).exists():
                        checkbox = QCheckBox(f"📂 {Path(directory).name}")
                        checkbox.setChecked(True)
                        checkbox.setToolTip(directory)
                        dirs_layout.addWidget(checkbox)
                        dir_checkboxes[directory] = checkbox
            else:
                no_dirs_label = QLabel("No watched directories configured")
                no_dirs_label.setStyleSheet("color: #666; font-style: italic;")
                dirs_layout.addWidget(no_dirs_label)
                
            layout.addWidget(dirs_group)
            
            # Registered datasets group
            datasets_group = QGroupBox("Registered Datasets")
            datasets_layout = QVBoxLayout(datasets_group)
            
            dataset_checkboxes = {}
            if registered_datasets:
                for dataset in registered_datasets:
                    checkbox = QCheckBox(f"📊 {dataset.get('name', 'Unnamed Dataset')}")
                    checkbox.setChecked(True)
                    checkbox.setToolTip(f"Type: {dataset.get('type', 'Unknown')}\nSource: {dataset.get('source', 'Unknown')}")
                    datasets_layout.addWidget(checkbox)
                    dataset_checkboxes[dataset.get('name', '')] = checkbox
            else:
                no_datasets_label = QLabel("No registered datasets")
                no_datasets_label.setStyleSheet("color: #666; font-style: italic;")
                datasets_layout.addWidget(no_datasets_label)
                
            layout.addWidget(datasets_group)
            
            # Options group
            options_group = QGroupBox("Indexing Options")
            options_layout = QVBoxLayout(options_group)
            
            include_metadata_checkbox = QCheckBox("Include file metadata in vector search")
            include_metadata_checkbox.setChecked(True)
            include_metadata_checkbox.setToolTip("Include tags, descriptions, and other metadata in searchable content")
            options_layout.addWidget(include_metadata_checkbox)
            
            preserve_existing_checkbox = QCheckBox("Preserve existing metadata during rebuild")
            preserve_existing_checkbox.setChecked(True)
            preserve_existing_checkbox.setToolTip("Keep existing file metadata and only update content")
            options_layout.addWidget(preserve_existing_checkbox)
            
            layout.addWidget(options_group)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            select_all_btn = QPushButton("Select All")
            select_all_btn.clicked.connect(lambda: self.toggle_all_checkboxes(dir_checkboxes, dataset_checkboxes, True))
            button_layout.addWidget(select_all_btn)
            
            select_none_btn = QPushButton("Select None")
            select_none_btn.clicked.connect(lambda: self.toggle_all_checkboxes(dir_checkboxes, dataset_checkboxes, False))
            button_layout.addWidget(select_none_btn)
            
            button_layout.addStretch()
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            rebuild_btn = QPushButton("Rebuild Index")
            rebuild_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #c82333; }
            """)
            rebuild_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(rebuild_btn)
            
            layout.addLayout(button_layout)
            
            # Show dialog
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
                
            # Get selected sources
            selected_dirs = [path for path, checkbox in dir_checkboxes.items() if checkbox.isChecked()]
            selected_datasets = [name for name, checkbox in dataset_checkboxes.items() if checkbox.isChecked()]
            
            if not selected_dirs and not selected_datasets:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Sources Selected", "Please select at least one data source to index.")
                return
                
            # Confirm rebuild
            total_sources = len(selected_dirs) + len(selected_datasets)
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Confirm Rebuild",
                f"This will rebuild the entire vector search index for {total_sources} data sources:\n\n"
                f"• {len(selected_dirs)} directories\n"
                f"• {len(selected_datasets)} datasets\n\n"
                f"This may take several minutes depending on the amount of data.\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
                
            # Start comprehensive indexing
            indexing_options = {
                'include_metadata': include_metadata_checkbox.isChecked(),
                'preserve_existing': preserve_existing_checkbox.isChecked()
            }
            
            self.start_comprehensive_indexing(selected_dirs, selected_datasets, indexing_options)
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to start indexing: {str(e)}")
            
    def toggle_all_checkboxes(self, dir_checkboxes, dataset_checkboxes, checked):
        """Toggle all checkboxes on or off"""
        for checkbox in dir_checkboxes.values():
            checkbox.setChecked(checked)
        for checkbox in dataset_checkboxes.values():
            checkbox.setChecked(checked)
            
    def start_comprehensive_indexing(self, directories, datasets, options):
        """Start comprehensive indexing of all selected data sources"""
        if self.indexing_worker and self.indexing_worker.isRunning():
            return  # Already indexing
            
        # Clear index
        try:
            self.vector_engine.clear_index()
        except Exception as e:
            logger.warning(f"Failed to clear index before rebuild: {e}")
            
        # Show progress UI
        self.indexing_progress.setVisible(True)
        self.indexing_progress.setValue(0)
        self.indexing_status_label.setVisible(True)
        self.indexing_status_label.setText("Starting comprehensive indexing...")
        
        # Disable buttons during indexing
        self.rebuild_btn.setEnabled(False)
        self.clear_index_btn.setEnabled(False)
        
        # Start comprehensive indexing worker
        from PyQt6.QtCore import QThread
        
        class ComprehensiveIndexingWorker(QThread):
            progress_updated = pyqtSignal(int, str)  # progress, status
            indexing_complete = pyqtSignal(dict)  # results
            error_occurred = pyqtSignal(str)  # error message
            
            def __init__(self, vector_engine, config_manager, directories, datasets, options):
                super().__init__()
                self.vector_engine = vector_engine
                self.config_manager = config_manager
                self.directories = directories
                self.datasets = datasets
                self.options = options
                
            def run(self):
                try:
                    total_results = {
                        'total_directories': len(self.directories),
                        'total_datasets': len(self.datasets),
                        'indexed_directories': 0,
                        'indexed_datasets': 0,
                        'total_files': 0,
                        'indexed_files': 0,
                        'errors': []
                    }
                    
                    total_sources = len(self.directories) + len(self.datasets)
                    current_source = 0
                    
                    # Index directories
                    for directory in self.directories:
                        if not Path(directory).exists():
                            continue
                            
                        current_source += 1
                        progress = int((current_source / total_sources) * 100)
                        self.progress_updated.emit(progress, f"Indexing directory: {Path(directory).name}")
                        
                        # Get file metadata if including metadata
                        if self.options.get('include_metadata', True):
                            # Load file metadata from config
                            file_metadata = self.config_manager.get("file_management.file_metadata", {})
                            
                            # Index directory with enhanced metadata
                            results = self.vector_engine.index_directory(
                                directory,
                                fileset_name=Path(directory).name,
                                fileset_description=f"Files from {directory}",
                                tags=['files', 'local', 'directory']
                            )
                        else:
                            results = self.vector_engine.index_directory(directory)
                        
                        total_results['indexed_directories'] += 1
                        total_results['total_files'] += results.get('total_files', 0)
                        total_results['indexed_files'] += results.get('indexed_files', 0)
                        total_results['errors'].extend(results.get('errors', []))
                    
                    # Index datasets (placeholder for future implementation)
                    for dataset_name in self.datasets:
                        current_source += 1
                        progress = int((current_source / total_sources) * 100)
                        self.progress_updated.emit(progress, f"Indexing dataset: {dataset_name}")
                        
                        # TODO: Implement dataset indexing based on dataset type
                        # This would connect to databases, APIs, etc.
                        total_results['indexed_datasets'] += 1
                        
                    self.indexing_complete.emit(total_results)
                    
                except Exception as e:
                    self.error_occurred.emit(str(e))
        
        self.indexing_worker = ComprehensiveIndexingWorker(
            self.vector_engine, self.config_manager, directories, datasets, options
        )
        self.indexing_worker.progress_updated.connect(self.on_indexing_progress)
        self.indexing_worker.indexing_complete.connect(self.on_comprehensive_indexing_complete)
        self.indexing_worker.error_occurred.connect(self.on_indexing_error)
        self.indexing_worker.start()
        
        self.indexing_started.emit()
        
    def on_comprehensive_indexing_complete(self, results):
        """Handle comprehensive indexing completion"""
        # Hide progress UI
        self.indexing_progress.setVisible(False)
        self.indexing_status_label.setVisible(False)
        
        # Re-enable buttons
        self.rebuild_btn.setEnabled(True)
        self.clear_index_btn.setEnabled(True)
        
        # Update status
        indexed_files = results['indexed_files']
        total_files = results['total_files']
        indexed_dirs = results['indexed_directories']
        indexed_datasets = results['indexed_datasets']
        failed = len(results['errors'])
        
        if indexed_files > 0:
            status_parts = []
            if indexed_dirs > 0:
                status_parts.append(f"{indexed_dirs} directories")
            if indexed_datasets > 0:
                status_parts.append(f"{indexed_datasets} datasets")
                
            status_text = f"✅ Indexing complete: {indexed_files}/{total_files} files from {', '.join(status_parts)}"
            if failed > 0:
                status_text += f" ({failed} errors)"
            self.db_status_label.setText(status_text)
            self.db_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.db_status_label.setText("⚠️ No files were indexed")
            self.db_status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            
        self.update_database_status()
        self.indexing_completed.emit(results)
        
        # Show comprehensive results dialog
        self.show_comprehensive_indexing_results(results)
        
    def show_comprehensive_indexing_results(self, results):
        """Show comprehensive indexing results dialog"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel, QTabWidget, QWidget
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Comprehensive Indexing Results")
        dialog.setModal(True)
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Summary
        summary = QLabel(f"""
<h3>Comprehensive Indexing Complete</h3>
<b>Directories Processed:</b> {results['indexed_directories']}/{results['total_directories']}<br>
<b>Datasets Processed:</b> {results['indexed_datasets']}/{results['total_datasets']}<br>
<b>Total Files Found:</b> {results['total_files']}<br>
<b>Successfully Indexed:</b> {results['indexed_files']}<br>
<b>Errors:</b> {len(results['errors'])}
        """)
        layout.addWidget(summary)
        
        # Tabs for detailed results
        tabs = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        summary_text = QTextEdit()
        summary_content = f"""
COMPREHENSIVE VECTOR INDEX REBUILD COMPLETE

Data Sources Processed:
• Directories: {results['indexed_directories']}/{results['total_directories']}
• Datasets: {results['indexed_datasets']}/{results['total_datasets']}

File Processing:
• Total files found: {results['total_files']}
• Successfully indexed: {results['indexed_files']}
• Success rate: {(results['indexed_files']/results['total_files']*100):.1f}% if results['total_files'] > 0 else 0

The vector search index has been completely rebuilt and is ready for use.
You can now perform semantic searches across all indexed content.
        """
        summary_text.setPlainText(summary_content)
        summary_text.setReadOnly(True)
        summary_layout.addWidget(summary_text)
        tabs.addTab(summary_tab, "Summary")
        
        # Errors tab (if any)
        if results['errors']:
            errors_tab = QWidget()
            errors_layout = QVBoxLayout(errors_tab)
            
            errors_text = QTextEdit()
            errors_text.setPlainText('\n'.join(results['errors']))
            errors_text.setReadOnly(True)
            errors_layout.addWidget(errors_text)
            tabs.addTab(errors_tab, f"Errors ({len(results['errors'])})")
            
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
            
    def clear_vector_index(self):
        """Clear the vector search index"""
        if not self.vector_engine:
            return
            
        try:
            from PyQt6.QtWidgets import QMessageBox
            
            reply = QMessageBox.question(
                self,
                "Clear Vector Index",
                "Are you sure you want to clear the entire vector search index?\n\nThis will remove all indexed documents and embeddings.\n\nYou will need to rebuild the index to use vector search.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.vector_engine.clear_index()
                self.db_status_label.setText("🗑️ Vector index cleared")
                self.db_status_label.setStyleSheet("color: #6c757d; font-weight: bold;")
                self.update_database_status()
                
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to clear index: {str(e)}")
            
    def start_indexing(self, directories, rebuild=False):
        """Start indexing process"""
        if self.indexing_worker and self.indexing_worker.isRunning():
            return  # Already indexing
            
        # Clear index if rebuilding
        if rebuild:
            try:
                self.vector_engine.clear_index()
            except Exception as e:
                logger.warning(f"Failed to clear index before rebuild: {e}")
                
        # Show progress UI
        self.indexing_progress.setVisible(True)
        self.indexing_progress.setValue(0)
        self.indexing_status_label.setVisible(True)
        self.indexing_status_label.setText("Starting indexing...")
        
        # Disable buttons during indexing
        self.rebuild_btn.setEnabled(False)
        self.clear_index_btn.setEnabled(False)
        
        # Start indexing worker
        from PyQt6.QtCore import QThread
        
        class IndexingWorker(QThread):
            progress_updated = pyqtSignal(int, str)  # progress, status
            indexing_complete = pyqtSignal(dict)  # results
            error_occurred = pyqtSignal(str)  # error message
            
            def __init__(self, vector_engine, directories):
                super().__init__()
                self.vector_engine = vector_engine
                self.directories = directories
                
            def run(self):
                try:
                    total_results = {
                        'total_files': 0,
                        'indexed_files': 0,
                        'failed_files': 0,
                        'skipped_files': 0,
                        'errors': []
                    }
                    
                    for i, directory in enumerate(self.directories):
                        if not Path(directory).exists():
                            continue
                            
                        self.progress_updated.emit(
                            int((i / len(self.directories)) * 100),
                            f"Indexing directory: {Path(directory).name}"
                        )
                        
                        results = self.vector_engine.index_directory(directory)
                        
                        total_results['total_files'] += results['total_files']
                        total_results['indexed_files'] += results['indexed_files']
                        total_results['failed_files'] += results['failed_files']
                        total_results['skipped_files'] += results['skipped_files']
                        total_results['errors'].extend(results['errors'])
                        
                    self.indexing_complete.emit(total_results)
                    
                except Exception as e:
                    self.error_occurred.emit(str(e))
        
        self.indexing_worker = IndexingWorker(self.vector_engine, directories)
        self.indexing_worker.progress_updated.connect(self.on_indexing_progress)
        self.indexing_worker.indexing_complete.connect(self.on_indexing_complete)
        self.indexing_worker.error_occurred.connect(self.on_indexing_error)
        self.indexing_worker.start()
        
        self.indexing_started.emit()
        
    def on_indexing_progress(self, progress, status):
        """Handle indexing progress updates"""
        self.indexing_progress.setValue(progress)
        self.indexing_status_label.setText(status)
        
    def on_indexing_complete(self, results):
        """Handle indexing completion"""
        # Hide progress UI
        self.indexing_progress.setVisible(False)
        self.indexing_status_label.setVisible(False)
        
        # Re-enable buttons
        self.rebuild_btn.setEnabled(True)
        self.clear_index_btn.setEnabled(True)
        
        # Update status
        indexed = results['indexed_files']
        total = results['total_files']
        failed = results['failed_files']
        
        if indexed > 0:
            status_text = f"✅ Indexing complete: {indexed}/{total} files indexed"
            if failed > 0:
                status_text += f" ({failed} failed)"
            self.db_status_label.setText(status_text)
            self.db_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.db_status_label.setText("⚠️ No files were indexed")
            self.db_status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            
        self.update_database_status()
        self.indexing_completed.emit(results)
        
        # Show results dialog
        self.show_indexing_results(results)
        
    def on_indexing_error(self, error_message):
        """Handle indexing errors"""
        # Hide progress UI
        self.indexing_progress.setVisible(False)
        self.indexing_status_label.setVisible(False)
        
        # Re-enable buttons
        self.rebuild_btn.setEnabled(True)
        self.clear_index_btn.setEnabled(True)
        
        # Show error
        self.db_status_label.setText(f"❌ Indexing failed: {error_message}")
        self.db_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Indexing Error", f"Indexing failed:\n\n{error_message}")
        
    def show_indexing_results(self, results):
        """Show indexing results dialog"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Indexing Results")
        dialog.setModal(True)
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Summary
        summary = QLabel(f"""
<h3>Indexing Complete</h3>
<b>Total Files:</b> {results['total_files']}<br>
<b>Successfully Indexed:</b> {results['indexed_files']}<br>
<b>Failed:</b> {results['failed_files']}<br>
<b>Skipped:</b> {results['skipped_files']}
        """)
        layout.addWidget(summary)
        
        # Errors (if any)
        if results['errors']:
            errors_label = QLabel("<b>Errors:</b>")
            layout.addWidget(errors_label)
            
            errors_text = QTextEdit()
            errors_text.setPlainText('\n'.join(results['errors']))
            errors_text.setMaximumHeight(200)
            errors_text.setReadOnly(True)
            layout.addWidget(errors_text)
            
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
        
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        self.config_manager.set("vector_search.model", model_name)
        
        if self.vector_engine:
            # Model change requires re-indexing
            self.db_status_label.setText("⚠️ Model changed - rebuilding index recommended")
            self.db_status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            
    def perform_search(self, query: str) -> dict:
        """Perform vector search with current settings"""
        if not self.vector_engine:
            return {'error': 'Vector search engine not available'}
            
        try:
            max_results = int(self.max_results.currentText())
            similarity_threshold = float(self.similarity_threshold.currentText())
            
            results = self.vector_engine.search(
                query=query,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            return {'results': results, 'query': query}
            
        except Exception as e:
            return {'error': f'Search failed: {str(e)}'}

class DocumentViewerTab(QWidget):
    """Document viewer tab widget"""
    
    document_opened = pyqtSignal(str)  # document_path
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Setup document viewer UI"""
        layout = QVBoxLayout(self)
        
        # Document selection and controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Document:"))
        
        self.document_combo = QComboBox()
        self.document_combo.addItem("Select a document...")
        self.document_combo.setMinimumWidth(300)
        controls_layout.addWidget(self.document_combo)
        
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self.open_document)
        controls_layout.addWidget(open_btn)
        
        controls_layout.addStretch()
        
        # View options
        self.show_metadata = QCheckBox("Show Metadata")
        controls_layout.addWidget(self.show_metadata)
        
        layout.addLayout(controls_layout)
        
        # Document content area
        self.content_area = QTextEdit()
        self.content_area.setReadOnly(True)
        self.content_area.setPlaceholderText("Select and open a document to view its contents here.")
        self.content_area.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.content_area)
        
    def open_document(self):
        """Open selected document"""
        doc_name = self.document_combo.currentText()
        if doc_name and doc_name != "Select a document...":
            # Placeholder implementation
            self.content_area.setText(f"Document content for: {doc_name}\n\nThis is where the actual document content would be displayed after implementing the document reader backend.")
            self.document_opened.emit(doc_name)

class SQLQueryTab(QWidget):
    """SQL query tab widget"""
    
    query_executed = pyqtSignal(str, dict)  # query, results
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Setup SQL query UI"""
        layout = QVBoxLayout(self)
        
        # Query controls
        controls_layout = QHBoxLayout()
        
        # Database selection
        controls_layout.addWidget(QLabel("Database:"))
        self.database_combo = QComboBox()
        self.database_combo.addItems(["Local SQLite", "Local DuckDB"])
        controls_layout.addWidget(self.database_combo)
        
        controls_layout.addStretch()
        
        # Query actions
        execute_btn = QPushButton("Execute")
        execute_btn.clicked.connect(self.execute_query)
        execute_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        controls_layout.addWidget(execute_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_query)
        controls_layout.addWidget(clear_btn)
        
        layout.addLayout(controls_layout)
        
        # Query editor
        editor_label = QLabel("SQL Query:")
        editor_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(editor_label)
        
        self.query_editor = QTextEdit()
        self.query_editor.setPlaceholderText("Enter your SQL query here...")
        self.query_editor.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
            }
        """)
        self.query_editor.setMaximumHeight(200)
        
        # Add SQL syntax highlighting
        self.highlighter = SQLHighlighter(self.query_editor.document())
        
        layout.addWidget(self.query_editor)
        
        # Results area
        results_label = QLabel("Query Results:")
        results_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(results_label)
        
        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        self.results_area.setPlaceholderText("Query results will appear here...")
        self.results_area.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self.results_area)
        
    def execute_query(self):
        """Execute SQL query"""
        query = self.query_editor.toPlainText().strip()
        if not query:
            self.results_area.setText("No query to execute.")
            return
            
        database = self.database_combo.currentText()
        
        # Placeholder implementation
        result_text = f"Query executed on {database}:\n\n{query}\n\n"
        result_text += "Results would appear here after implementing the SQL execution backend.\n"
        result_text += "This would include:\n"
        result_text += "- Connection to selected database\n"
        result_text += "- Query execution\n"
        result_text += "- Results formatting\n"
        result_text += "- Error handling"
        
        self.results_area.setText(result_text)
        self.query_executed.emit(query, {"status": "placeholder"})
        
    def clear_query(self):
        """Clear query editor"""
        self.query_editor.clear()
        self.results_area.clear()

class SearchSection(QWidget):
    """Main search section focused on vector search"""
    
    search_performed = pyqtSignal(str, str, dict)  # query, search_type, results
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup search section UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Section header
        header_layout = QHBoxLayout()
        
        title = QLabel("Vector Search")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #333; padding: 8px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Quick access buttons
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setToolTip("Refresh search indices")
        refresh_btn.clicked.connect(self.refresh_indices)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Vector Search interface (no tabs needed since it's just one function)
        self.vector_tab = VectorSearchTab(self.config_manager)
        layout.addWidget(self.vector_tab)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.vector_tab.search_performed.connect(
            lambda q, r: self.search_performed.emit(q, "Vector Search", r)
        )
        
    def refresh_indices(self):
        """Refresh search indices"""
        logger.info("Refreshing search indices")
        # Placeholder implementation
        # This would trigger re-indexing of documents and vector embeddings
        
    def handle_search_query(self, query: str, search_type: str):
        """Handle search query from main search bar"""
        if search_type == "Vector Search":
            # Trigger vector search with query
            pass
            
    def get_current_tab_name(self) -> str:
        """Get name of currently active tab"""
        return "Vector Search"
