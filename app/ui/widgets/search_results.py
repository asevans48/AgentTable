"""
Search Results Widget
Displays search results in Google-like format with summaries
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
    QFrame, QPushButton, QProgressBar, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QCursor
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SearchResultItem(QFrame):
    """Individual search result item widget"""
    
    item_clicked = pyqtSignal(dict)  # result_data
    chat_requested = pyqtSignal(dict)  # result_data
    selection_changed = pyqtSignal(dict, bool)  # result_data, is_selected
    
    def __init__(self, result_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.result_data = result_data
        self.is_selected = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the result item UI"""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin: 4px;
                padding: 8px;
            }
            QFrame:hover {
                border-color: #1a73e8;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Top row with checkbox and title
        top_layout = QHBoxLayout()
        
        # Selection checkbox
        self.selection_checkbox = QCheckBox()
        self.selection_checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #ddd;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #1a73e8;
                border-color: #1a73e8;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0LjUgOEwxMSAxIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }
            QCheckBox::indicator:hover {
                border-color: #1a73e8;
            }
        """)
        self.selection_checkbox.stateChanged.connect(self.on_selection_changed)
        top_layout.addWidget(self.selection_checkbox)
        
        # Header row with title and metadata
        header_layout = QHBoxLayout()
        
        # Title
        title = self.result_data.get('title', 'Untitled')
        self.title_label = QLabel(f"<h3 style='color: #1a73e8; margin: 0;'>{title}</h3>")
        self.title_label.setWordWrap(True)
        self.title_label.mousePressEvent = self.on_title_clicked
        self.title_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        header_layout.addWidget(self.title_label, 1)
        
        # Metadata badges
        metadata_layout = QVBoxLayout()
        metadata_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Source type badge
        source_type = self.result_data.get('source_type', 'Unknown')
        type_badge = QLabel(source_type)
        type_badge.setStyleSheet("""
            QLabel {
                background-color: #e8f0fe;
                color: #1a73e8;
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 10pt;
                font-weight: bold;
            }
        """)
        type_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metadata_layout.addWidget(type_badge)
        
        # Access badge
        access_level = self.result_data.get('access_level', 'Unknown')
        access_color = '#4caf50' if access_level == 'Full' else '#ff9800'
        access_badge = QLabel(access_level)
        access_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {access_color}20;
                color: {access_color};
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 9pt;
            }}
        """)
        access_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metadata_layout.addWidget(access_badge)
        
        header_layout.addLayout(metadata_layout)
        top_layout.addLayout(header_layout, 1)
        layout.addLayout(top_layout)
        
        # URL/Path (if available)
        source_path = self.result_data.get('source_path', '')
        if source_path:
            path_label = QLabel(f"<span style='color: #006621; font-size: 10pt;'>{source_path}</span>")
            path_label.setWordWrap(True)
            layout.addWidget(path_label)
        
        # Metadata description (if available and different from summary)
        user_description = self.result_data.get('user_description', '')
        if user_description and user_description.strip():
            desc_label = QLabel(f"<b>Description:</b> {user_description}")
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("""
                QLabel {
                    color: #1a73e8;
                    font-size: 10pt;
                    margin: 2px 0;
                    padding: 4px 8px;
                    background-color: #f0f7ff;
                    border-left: 3px solid #1a73e8;
                    border-radius: 4px;
                }
            """)
            layout.addWidget(desc_label)

        # Tags display
        tags = self.result_data.get('tags', [])
        if tags and isinstance(tags, list) and any(tag.strip() for tag in tags):
            tags_widget = QWidget()
            tags_layout = QHBoxLayout(tags_widget)
            tags_layout.setContentsMargins(0, 2, 0, 2)
            
            tags_title = QLabel("Tags:")
            tags_title.setStyleSheet("color: #666; font-size: 9pt; font-weight: bold;")
            tags_layout.addWidget(tags_title)
            
            # Display up to 5 tags as badges
            for tag in tags[:5]:
                if tag.strip():
                    tag_badge = QLabel(tag.strip())
                    tag_badge.setStyleSheet("""
                        QLabel {
                            background-color: #e8f5e8;
                            color: #2e7d32;
                            padding: 2px 6px;
                            border-radius: 8px;
                            font-size: 8pt;
                            font-weight: bold;
                            margin: 1px;
                        }
                    """)
                    tags_layout.addWidget(tag_badge)
            
            # Show count if more tags exist
            if len(tags) > 5:
                more_label = QLabel(f"+{len(tags) - 5} more")
                more_label.setStyleSheet("color: #666; font-size: 8pt; font-style: italic;")
                tags_layout.addWidget(more_label)
                
            tags_layout.addStretch()
            layout.addWidget(tags_widget)

        # Summary/Content preview
        summary = self.result_data.get('summary', self.result_data.get('description', ''))
        if summary:
            # Clean up the summary to remove redundant metadata that's now shown separately
            clean_summary = summary
            if user_description and user_description in clean_summary:
                clean_summary = clean_summary.replace(f"📝 {user_description}", "").strip()
            
            # Remove leading newlines and clean up
            clean_summary = clean_summary.strip()
            if clean_summary.startswith('\n'):
                clean_summary = clean_summary.lstrip('\n')
            
            summary_label = QLabel(clean_summary)
            summary_label.setWordWrap(True)
            summary_label.setStyleSheet("""
                QLabel {
                    color: #333;
                    font-size: 11pt;
                    line-height: 1.4;
                    margin: 4px 0;
                }
            """)
            layout.addWidget(summary_label)
        
        # Schema information (if available)
        schema_info = self.result_data.get('schema_info', '')
        if schema_info and schema_info.strip():
            schema_label = QLabel(f"<b>Schema:</b> {schema_info}")
            schema_label.setWordWrap(True)
            schema_label.setStyleSheet("""
                QLabel {
                    color: #6a1b9a;
                    font-size: 9pt;
                    margin: 2px 0;
                    padding: 3px 6px;
                    background-color: #f3e5f5;
                    border-radius: 3px;
                }
            """)
            layout.addWidget(schema_label)

        # Footer with actions and metadata
        footer_layout = QHBoxLayout()
        
        # Dataset and metadata information
        fileset_name = self.result_data.get('fileset_name', self.result_data.get('owner', 'Unknown'))
        owner_label = QLabel(f"📊 {fileset_name}")
        owner_label.setStyleSheet("color: #666; font-size: 9pt; font-weight: bold;")
        footer_layout.addWidget(owner_label)
        
        # File type information
        file_type = self.result_data.get('file_type', '')
        if file_type:
            type_label = QLabel(f"📄 {file_type.upper()}")
            type_label.setStyleSheet("color: #666; font-size: 8pt;")
            footer_layout.addWidget(type_label)
        
        # Similarity score (for vector search results)
        score = self.result_data.get('score', 0)
        if score > 0:
            score_label = QLabel(f"🎯 {score:.2f}")
            score_label.setToolTip(f"Relevance score: {score:.3f}")
            score_label.setStyleSheet("color: #666; font-size: 8pt;")
            footer_layout.addWidget(score_label)
        
        # Last modified
        last_modified = self.result_data.get('last_modified', '')
        if last_modified:
            modified_label = QLabel(f"🕒 {last_modified}")
            modified_label.setStyleSheet("color: #666; font-size: 8pt;")
            footer_layout.addWidget(modified_label)
        
        footer_layout.addStretch()
        
        # Action buttons
        if self.result_data.get('can_chat', False):
            chat_button = QPushButton("Chat")
            chat_button.setStyleSheet("""
                QPushButton {
                    background-color: #1a73e8;
                    color: white;
                    border: none;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 9pt;
                }
                QPushButton:hover {
                    background-color: #1557b0;
                }
            """)
            chat_button.clicked.connect(self.on_chat_clicked)
            footer_layout.addWidget(chat_button)
        
        view_button = QPushButton("View")
        view_button.setStyleSheet("""
            QPushButton {
                background-color: #34a853;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #2d8f47;
            }
        """)
        view_button.clicked.connect(self.on_view_clicked)
        footer_layout.addWidget(view_button)
        
        layout.addLayout(footer_layout)
        
    def on_selection_changed(self, state):
        """Handle checkbox selection change"""
        self.is_selected = state == Qt.CheckState.Checked.value
        self.selection_changed.emit(self.result_data, self.is_selected)
        
        # Update visual appearance based on selection
        if self.is_selected:
            self.setStyleSheet("""
                QFrame {
                    background-color: #e3f2fd;
                    border: 2px solid #1a73e8;
                    border-radius: 8px;
                    margin: 4px;
                    padding: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    margin: 4px;
                    padding: 8px;
                }
                QFrame:hover {
                    border-color: #1a73e8;
                    background-color: #f8f9fa;
                }
            """)
    
    def set_selected(self, selected: bool):
        """Programmatically set selection state"""
        self.selection_checkbox.setChecked(selected)
    
    def get_selection_data(self) -> Dict[str, Any]:
        """Get data for tracking this selection"""
        return {
            'id': self.get_unique_id(),
            'type': 'dataset' if self.result_data.get('is_dataset', False) else 'file',
            'name': self.result_data.get('title', 'Unknown'),
            'path': self.result_data.get('source_path', ''),
            'fileset_name': self.result_data.get('fileset_name', ''),
            'result_data': self.result_data
        }
    
    def get_unique_id(self) -> str:
        """Generate unique ID for this result item"""
        # Use path as primary identifier, fallback to title + type
        path = self.result_data.get('source_path', '')
        if path:
            return path
        else:
            title = self.result_data.get('title', 'Unknown')
            item_type = self.result_data.get('source_type', 'Unknown')
            return f"{item_type}:{title}"
        
    def on_title_clicked(self, event):
        """Handle title click"""
        self.item_clicked.emit(self.result_data)
        
    def on_chat_clicked(self):
        """Handle chat button click"""
        self.chat_requested.emit(self.result_data)
        
    def on_view_clicked(self):
        """Handle view button click"""
        self.item_clicked.emit(self.result_data)

class SearchWorker(QThread):
    """Background worker for performing searches"""
    
    results_ready = pyqtSignal(list)  # search results
    error_occurred = pyqtSignal(str)  # error message
    progress_updated = pyqtSignal(int)  # progress percentage
    
    def __init__(self, query: str, search_type: str, config_manager):
        super().__init__()
        self.query = query
        self.search_type = search_type
        self.config_manager = config_manager
        
    def run(self):
        """Run the search in background"""
        try:
            self.progress_updated.emit(10)
            
            if self.search_type == "Vector Search":
                results = self.perform_vector_search()
            elif self.search_type == "Dataset Search":
                results = self.search_datasets()
            elif self.search_type == "Document Search":
                results = self.search_documents()
            elif self.search_type == "SQL Query":
                results = self.execute_sql_query()
            else:
                results = self.perform_general_search()
                
            self.progress_updated.emit(100)
            self.results_ready.emit(results)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.error_occurred.emit(str(e))
            
    def perform_vector_search(self) -> List[Dict[str, Any]]:
        """Perform vector search using the vector search engine"""
        try:
            from utils.vector_search import VectorSearchEngine
            
            vector_engine = VectorSearchEngine(self.config_manager)
            
            # First test the search functionality
            test_result = vector_engine.test_search_functionality()
            if test_result['status'] == 'failed':
                return [{
                    'title': 'Vector Search Setup Error',
                    'source_type': 'Error',
                    'source_path': 'system',
                    'summary': f"Vector search is not properly configured: {test_result.get('error', 'Unknown error')}",
                    'owner': 'System',
                    'last_modified': '',
                    'access_level': 'Full',
                    'can_chat': False,
                    'score': 0.0
                }]
            
            # Perform the actual search
            results = vector_engine.search(
                self.query, 
                max_results=20,
                similarity_threshold=0.2  # Lower threshold to get more results
            )
            
            # Transform results to match expected format
            transformed_results = []
            for result in results:
                if 'error' in result:
                    # Show detailed error information
                    error_summary = result.get('message', 'Unknown error')
                    if 'debug_info' in result:
                        debug_info = result['debug_info']
                        error_summary += f"\n\nDebug Info:\n"
                        error_summary += f"• Processed chunks: {debug_info.get('processed_chunks', 0)}\n"
                        error_summary += f"• Missing embeddings: {debug_info.get('missing_embeddings', 0)}\n"
                        error_summary += f"• Similarity threshold: {debug_info.get('similarity_threshold', 0.3)}\n"
                        error_summary += f"• Database: {debug_info.get('database_path', 'Unknown')}"
                    
                    transformed_results.append({
                        'title': 'Vector Search Error',
                        'source_type': 'Error',
                        'source_path': 'system',
                        'summary': error_summary,
                        'owner': 'System',
                        'last_modified': '',
                        'access_level': 'Full',
                        'can_chat': False,
                        'score': 0.0
                    })
                elif 'message' in result:
                    # Show informational messages with debug details
                    info_summary = result['message']
                    if 'suggestion' in result:
                        info_summary += f"\n\nSuggestion: {result['suggestion']}"
                    if 'debug_info' in result:
                        debug_info = result['debug_info']
                        info_summary += f"\n\nDebug Info:\n"
                        info_summary += f"• Processed chunks: {debug_info.get('processed_chunks', 0)}\n"
                        info_summary += f"• Missing embeddings: {debug_info.get('missing_embeddings', 0)}\n"
                        info_summary += f"• Similarity threshold: {debug_info.get('similarity_threshold', 0.3)}"
                    if 'stats' in result:
                        stats = result['stats']
                        info_summary += f"\n\nDatabase Stats:\n"
                        info_summary += f"• Total documents: {stats.get('total_documents', 0)}\n"
                        info_summary += f"• Indexed documents: {stats.get('indexed_documents', 0)}\n"
                        info_summary += f"• Total chunks: {stats.get('total_chunks', 0)}"
                    
                    transformed_results.append({
                        'title': 'Vector Search Info',
                        'source_type': 'Info',
                        'source_path': 'system',
                        'summary': info_summary,
                        'owner': 'System',
                        'last_modified': '',
                        'access_level': 'Full',
                        'can_chat': False,
                        'score': 0.0
                    })
                else:
                    # Format content preview
                    content = result['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                        
                    # Get file name from path
                    from pathlib import Path
                    file_name = Path(result.get('file_path', 'Unknown')).name
                        
                    # Get enhanced metadata from vector search result
                    fileset_name = result.get('fileset_name', 'Unknown Dataset')
                    schema_info = result.get('schema_info', '')
                    tags = result.get('tags', '').split(',') if result.get('tags') else []
                    user_description = result.get('user_description', '')
                    
                    # Create enhanced summary with metadata context
                    summary_parts = []
                    if user_description:
                        summary_parts.append(f"📝 {user_description}")
                    if schema_info:
                        summary_parts.append(f"🏗️ {schema_info}")
                    summary_parts.append(content)
                    enhanced_summary = '\n'.join(summary_parts)
                    
                    # Add enhanced scoring information to summary
                    similarity_score = result.get('similarity', 0.0)
                    enhanced_summary += f"\n\n🎯 Overall Score: {similarity_score:.3f}"
                    
                    # Show component scores if available
                    if 'vector_score' in result:
                        enhanced_summary += f"\n📊 Vector: {result['vector_score']:.3f}"
                    if 'keyword_score' in result:
                        enhanced_summary += f" | Keyword: {result['keyword_score']:.3f}"
                    if 'metadata_score' in result:
                        enhanced_summary += f" | Metadata: {result['metadata_score']:.3f}"
                    
                    # Determine if this is a dataset or file
                    is_dataset = result.get('file_path', '').startswith('dataset://')
                    source_type = 'Dataset' if is_dataset else 'Document'
                    
                    transformed_results.append({
                        'title': result.get('title', 'Untitled'),
                        'source_type': source_type,
                        'source_path': result.get('file_path', 'Unknown'),
                        'summary': enhanced_summary,
                        'owner': fileset_name,  # Use fileset name as owner
                        'last_modified': '',
                        'access_level': 'Full',
                        'can_chat': True,
                        'score': similarity_score,
                        'file_type': result.get('file_type', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'document_id': result.get('document_id'),
                        'fileset_name': fileset_name,
                        'schema_info': schema_info,
                        'tags': tags,
                        'user_description': user_description,
                        'is_dataset': is_dataset
                    })
                    
            return transformed_results
            
        except ImportError:
            return [{
                'title': 'Vector Search Unavailable',
                'source_type': 'Error',
                'source_path': 'system',
                'summary': 'Vector search dependencies not installed. Please install: pip install sentence-transformers faiss-cpu numpy',
                'owner': 'System',
                'last_modified': '',
                'access_level': 'Full',
                'can_chat': False,
                'score': 0.0
            }]
        except Exception as e:
            import traceback
            return [{
                'title': 'Vector Search Error',
                'source_type': 'Error',
                'source_path': 'system',
                'summary': f'Error performing vector search: {str(e)}\n\nTraceback:\n{traceback.format_exc()}',
                'owner': 'System',
                'last_modified': '',
                'access_level': 'Full',
                'can_chat': False,
                'score': 0.0
            }]
        
    def search_datasets(self) -> List[Dict[str, Any]]:
        """Search registered datasets"""
        datasets = self.config_manager.get_registered_datasets()
        # Filter datasets based on query
        # This is a placeholder implementation
        return []
        
    def search_documents(self) -> List[Dict[str, Any]]:
        """Search documents in watched directories"""
        # Placeholder implementation
        return []
        
    def execute_sql_query(self) -> List[Dict[str, Any]]:
        """Execute SQL query"""
        # Placeholder implementation
        return []
        
    def perform_general_search(self) -> List[Dict[str, Any]]:
        """Perform general search across all sources"""
        # Placeholder implementation
        return []
        

class SearchResults(QWidget):
    """Search results display widget"""
    
    item_selected = pyqtSignal(dict)  # selected result data
    chat_requested = pyqtSignal(dict)  # result data for chat
    selection_changed = pyqtSignal(list)  # list of selected items
    
    def __init__(self, config_manager=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.current_results = []
        self.search_worker = None
        self.selected_items = {}  # Track selected items by unique ID
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the search results UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Search status and controls
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Enter a search query to get started")
        self.status_label.setStyleSheet("color: #666; font-size: 11pt; padding: 8px;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Selection controls
        self.selection_controls = QWidget()
        self.selection_controls.setVisible(False)
        selection_layout = QHBoxLayout(self.selection_controls)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        
        self.selection_label = QLabel("0 selected")
        self.selection_label.setStyleSheet("color: #1a73e8; font-weight: bold; font-size: 9pt;")
        selection_layout.addWidget(self.selection_label)
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #1a73e8;
                color: #1a73e8;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
            }
        """)
        self.select_all_btn.clicked.connect(self.select_all_items)
        selection_layout.addWidget(self.select_all_btn)
        
        self.clear_selection_btn = QPushButton("Clear")
        self.clear_selection_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #666;
                color: #666;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #f5f5f5;
            }
        """)
        self.clear_selection_btn.clicked.connect(self.clear_all_selections)
        selection_layout.addWidget(self.clear_selection_btn)
        
        status_layout.addWidget(self.selection_controls)
        
        # Sort and filter controls
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Relevance", "Date", "Owner", "Type"])
        self.sort_combo.setVisible(False)
        status_layout.addWidget(self.sort_combo)
        
        layout.addLayout(status_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #1a73e8;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Results area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.results_scroll.setWidget(self.results_container)
        layout.addWidget(self.results_scroll)
        
        # Welcome message
        self.show_welcome_message()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.sort_combo.currentTextChanged.connect(self.sort_results)
        
    def show_welcome_message(self):
        """Show welcome message when no search has been performed"""
        welcome_widget = QFrame()
        welcome_widget.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 20px;
                margin: 20px;
            }
        """)
        
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("Welcome to Data Platform Search")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #333; margin-bottom: 10px;")
        welcome_layout.addWidget(title)
        
        subtitle = QLabel("Search across all your data sources using:")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; margin-bottom: 15px;")
        welcome_layout.addWidget(subtitle)
        
        features = [
            "🔍 Vector Search - Find semantically similar content",
            "💬 AI Chat - Ask questions about your data",
            "📊 Dataset Search - Find specific datasets",
            "📄 Document Search - Search through documents",
            "🗄️ SQL Query - Query databases directly"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setFont(QFont("Arial", 11))
            feature_label.setStyleSheet("color: #555; margin: 2px 0; padding-left: 20px;")
            welcome_layout.addWidget(feature_label)
        
        self.results_layout.addWidget(welcome_widget)
        
    def perform_search(self, query: str, search_type: str):
        """Perform search with given query and type"""
        if not query.strip():
            return
            
        # Clear previous results and selections
        self.clear_results()
        self.clear_all_selections()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Searching for: {query}")
        self.sort_combo.setVisible(False)
        self.selection_controls.setVisible(False)
        
        # Start background search
        self.search_worker = SearchWorker(query, search_type, self.config_manager)
        self.search_worker.results_ready.connect(self.on_results_ready)
        self.search_worker.error_occurred.connect(self.on_search_error)
        self.search_worker.progress_updated.connect(self.progress_bar.setValue)
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.start()
        
    def on_results_ready(self, results: List[Dict[str, Any]]):
        """Handle search results"""
        self.current_results = results
        self.display_results(results)
        
    def on_search_error(self, error_message: str):
        """Handle search errors"""
        self.status_label.setText(f"Search error: {error_message}")
        self.show_error_message(error_message)
        
    def on_search_finished(self):
        """Handle search completion"""
        self.progress_bar.setVisible(False)
        if self.current_results:
            self.status_label.setText(f"Found {len(self.current_results)} results")
            self.sort_combo.setVisible(True)
            self.selection_controls.setVisible(True)
        else:
            self.status_label.setText("No results found")
            self.show_no_results_message()
        
    def display_results(self, results: List[Dict[str, Any]]):
        """Display search results"""
        self.clear_results()
        
        for result in results:
            result_item = SearchResultItem(result)
            result_item.item_clicked.connect(self.item_selected.emit)
            result_item.chat_requested.connect(self.chat_requested.emit)
            result_item.selection_changed.connect(self.on_item_selection_changed)
            self.results_layout.addWidget(result_item)
            
        # Add stretch to push results to top
        self.results_layout.addStretch()
        
    def clear_results(self):
        """Clear current results"""
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def show_no_results_message(self):
        """Show message when no results found"""
        no_results_widget = QFrame()
        no_results_widget.setStyleSheet("""
            QFrame {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 20px;
                margin: 20px;
            }
        """)
        
        layout = QVBoxLayout(no_results_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("No results found")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        suggestions = QLabel("Try:\n• Different keywords\n• Broader search terms\n• Checking your access permissions")
        suggestions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        suggestions.setStyleSheet("color: #856404;")
        layout.addWidget(suggestions)
        
        self.results_layout.addWidget(no_results_widget)
        
    def show_error_message(self, error: str):
        """Show error message"""
        error_widget = QFrame()
        error_widget.setStyleSheet("""
            QFrame {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 8px;
                padding: 20px;
                margin: 20px;
            }
        """)
        
        layout = QVBoxLayout(error_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("Search Error")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #721c24;")
        layout.addWidget(title)
        
        error_label = QLabel(error)
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        error_label.setStyleSheet("color: #721c24;")
        error_label.setWordWrap(True)
        layout.addWidget(error_label)
        
        self.results_layout.addWidget(error_widget)
        
    def on_item_selection_changed(self, result_data: Dict[str, Any], is_selected: bool):
        """Handle individual item selection change"""
        # Find the result item that changed
        result_item = None
        for i in range(self.results_layout.count()):
            widget = self.results_layout.itemAt(i).widget()
            if isinstance(widget, SearchResultItem) and widget.result_data == result_data:
                result_item = widget
                break
        
        if not result_item:
            return
            
        selection_data = result_item.get_selection_data()
        unique_id = selection_data['id']
        
        if is_selected:
            self.selected_items[unique_id] = selection_data
        else:
            self.selected_items.pop(unique_id, None)
            
        self.update_selection_display()
        self.selection_changed.emit(list(self.selected_items.values()))
    
    def update_selection_display(self):
        """Update the selection count display"""
        count = len(self.selected_items)
        if count == 0:
            self.selection_label.setText("0 selected")
        elif count == 1:
            self.selection_label.setText("1 item selected")
        else:
            self.selection_label.setText(f"{count} items selected")
    
    def select_all_items(self):
        """Select all visible result items"""
        for i in range(self.results_layout.count()):
            widget = self.results_layout.itemAt(i).widget()
            if isinstance(widget, SearchResultItem):
                widget.set_selected(True)
    
    def clear_all_selections(self):
        """Clear all selections"""
        self.selected_items.clear()
        
        # Update UI
        for i in range(self.results_layout.count()):
            widget = self.results_layout.itemAt(i).widget()
            if isinstance(widget, SearchResultItem):
                widget.set_selected(False)
                
        self.update_selection_display()
        self.selection_changed.emit([])
    
    def get_selected_items(self) -> List[Dict[str, Any]]:
        """Get list of currently selected items"""
        return list(self.selected_items.values())
    
    def get_selected_files(self) -> List[Dict[str, Any]]:
        """Get list of selected files only"""
        return [item for item in self.selected_items.values() if item['type'] == 'file']
    
    def get_selected_datasets(self) -> List[Dict[str, Any]]:
        """Get list of selected datasets only"""
        return [item for item in self.selected_items.values() if item['type'] == 'dataset']
    
    def sort_results(self, sort_by: str):
        """Sort results by given criteria"""
        if not self.current_results:
            return
            
        if sort_by == "Relevance":
            sorted_results = sorted(self.current_results, key=lambda x: x.get('score', 0), reverse=True)
        elif sort_by == "Date":
            sorted_results = sorted(self.current_results, key=lambda x: x.get('last_modified', ''), reverse=True)
        elif sort_by == "Owner":
            sorted_results = sorted(self.current_results, key=lambda x: x.get('owner', ''))
        elif sort_by == "Type":
            sorted_results = sorted(self.current_results, key=lambda x: x.get('source_type', ''))
        else:
            sorted_results = self.current_results
            
        self.display_results(sorted_results)
