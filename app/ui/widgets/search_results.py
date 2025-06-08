"""
Search Results Widget
Displays search results in Google-like format with summaries
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
    QFrame, QPushButton, QProgressBar, QComboBox
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
    
    def __init__(self, result_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.result_data = result_data
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
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
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
        layout.addLayout(header_layout)
        
        # URL/Path (if available)
        source_path = self.result_data.get('source_path', '')
        if source_path:
            path_label = QLabel(f"<span style='color: #006621; font-size: 10pt;'>{source_path}</span>")
            path_label.setWordWrap(True)
            layout.addWidget(path_label)
        
        # Summary/Description
        summary = self.result_data.get('summary', self.result_data.get('description', ''))
        if summary:
            summary_label = QLabel(summary)
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
        
        # Footer with actions and metadata
        footer_layout = QHBoxLayout()
        
        # Owner information
        owner = self.result_data.get('owner', 'Unknown')
        owner_label = QLabel(f"Owner: {owner}")
        owner_label.setStyleSheet("color: #666; font-size: 9pt;")
        footer_layout.addWidget(owner_label)
        
        # Last modified
        last_modified = self.result_data.get('last_modified', '')
        if last_modified:
            modified_label = QLabel(f"Modified: {last_modified}")
            modified_label.setStyleSheet("color: #666; font-size: 9pt;")
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
        """Perform vector search (placeholder implementation)"""
        import time
        time.sleep(1)  # Simulate processing
        
        # Mock results for demonstration
        return [
            {
                'title': 'Customer Data Analysis Report',
                'source_type': 'Document',
                'source_path': '/documents/reports/customer_analysis_2024.pdf',
                'summary': 'Comprehensive analysis of customer behavior patterns and preferences based on 2024 data. Includes demographic breakdowns, purchase patterns, and recommendations for marketing strategies.',
                'owner': 'Data Team',
                'last_modified': '2024-12-15',
                'access_level': 'Full',
                'can_chat': True,
                'score': 0.95
            },
            {
                'title': 'Sales Performance Dataset',
                'source_type': 'Dataset',
                'source_path': 'bigquery://project.dataset.sales_performance',
                'summary': 'Monthly sales performance data including revenue, units sold, and regional breakdowns. Updated daily with latest transaction data.',
                'owner': 'Sales Team',
                'last_modified': '2024-12-20',
                'access_level': 'Read-Only',
                'can_chat': True,
                'score': 0.87
            },
            {
                'title': 'Market Research Survey Results',
                'source_type': 'Spreadsheet',
                'source_path': '/shared/surveys/market_research_q4_2024.xlsx',
                'summary': 'Q4 2024 market research survey responses from 1,200 participants. Includes satisfaction scores, feature requests, and competitive analysis.',
                'owner': 'Marketing Team',
                'last_modified': '2024-12-10',
                'access_level': 'Full',
                'can_chat': False,
                'score': 0.79
            }
        ]
        
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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_results = []
        self.search_worker = None
        
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
            
        # Clear previous results
        self.clear_results()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Searching for: {query}")
        self.sort_combo.setVisible(False)
        
        # Start background search
        # Note: In real implementation, you'd pass actual config_manager
        self.search_worker = SearchWorker(query, search_type, None)
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
