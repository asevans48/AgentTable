"""
Search Section Widget
Main search interface with vector search, document viewer, and SQL query tabs
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit,
    QLabel, QPushButton, QFrame, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor
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
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Setup vector search UI"""
        layout = QVBoxLayout(self)
        
        # Search configuration
        config_frame = QFrame()
        config_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        config_layout = QHBoxLayout(config_frame)
        
        # Embedding model selection
        config_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "text-embedding-ada-002"
        ])
        config_layout.addWidget(self.model_combo)
        
        # Search scope
        config_layout.addWidget(QLabel("Scope:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["All Documents", "Recent Files", "Accessible Only"])
        config_layout.addWidget(self.scope_combo)
        
        # Max results
        config_layout.addWidget(QLabel("Max Results:"))
        self.max_results = QComboBox()
        self.max_results.addItems(["10", "25", "50", "100"])
        config_layout.addWidget(self.max_results)
        
        config_layout.addStretch()
        
        # Semantic search toggle
        self.semantic_search = QCheckBox("Semantic Search")
        self.semantic_search.setChecked(True)
        config_layout.addWidget(self.semantic_search)
        
        layout.addWidget(config_frame)
        
        # Results area (placeholder)
        results_label = QLabel("Vector search results will appear here after implementing the search backend.")
        results_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-style: italic;
                text-align: center;
                padding: 20px;
                border: 2px dashed #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
            }
        """)
        results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_label.setWordWrap(True)
        layout.addWidget(results_label)

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
    """Main search section with tabs for different search types"""
    
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
        
        title = QLabel("Search & Query")
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
        
        # Tab widget for different search types
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
        """)
        
        # Vector Search tab
        self.vector_tab = VectorSearchTab(self.config_manager)
        self.tab_widget.addTab(self.vector_tab, "🔍 Vector Search")
        
        # Document Viewer tab
        self.document_tab = DocumentViewerTab(self.config_manager)
        self.tab_widget.addTab(self.document_tab, "📄 Document Viewer")
        
        # SQL Query tab
        self.sql_tab = SQLQueryTab(self.config_manager)
        self.tab_widget.addTab(self.sql_tab, "🗄️ SQL Query")
        
        layout.addWidget(self.tab_widget)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.vector_tab.search_performed.connect(
            lambda q, r: self.search_performed.emit(q, "Vector Search", r)
        )
        self.document_tab.document_opened.connect(
            lambda d: self.search_performed.emit(d, "Document Viewer", {"document": d})
        )
        self.sql_tab.query_executed.connect(
            lambda q, r: self.search_performed.emit(q, "SQL Query", r)
        )
        
    def refresh_indices(self):
        """Refresh search indices"""
        logger.info("Refreshing search indices")
        # Placeholder implementation
        # This would trigger re-indexing of documents and vector embeddings
        
    def handle_search_query(self, query: str, search_type: str):
        """Handle search query from main search bar"""
        if search_type == "Vector Search":
            self.tab_widget.setCurrentWidget(self.vector_tab)
            # Trigger vector search with query
        elif search_type == "SQL Query":
            self.tab_widget.setCurrentWidget(self.sql_tab)
            self.sql_tab.query_editor.setText(query)
        elif search_type == "Document Search":
            self.tab_widget.setCurrentWidget(self.document_tab)
            # Trigger document search with query
            
    def get_current_tab_name(self) -> str:
        """Get name of currently active tab"""
        current_index = self.tab_widget.currentIndex()
        return self.tab_widget.tabText(current_index)
        
    def switch_to_tab(self, tab_name: str):
        """Switch to specific tab by name"""
        for i in range(self.tab_widget.count()):
            if tab_name.lower() in self.tab_widget.tabText(i).lower():
                self.tab_widget.setCurrentIndex(i)
                break
