"""
Analytics Section Widget
Charts & Reports, SQL Query, and Document Viewer functionality
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit,
    QLabel, QPushButton, QFrame, QComboBox, QCheckBox, QSplitter, 
    QGroupBox, QFormLayout
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

class ChartsReportsTab(QWidget):
    """Charts and reports creation tab"""
    
    chart_created = pyqtSignal(dict)  # chart_config
    report_generated = pyqtSignal(dict)  # report_config
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Setup charts and reports UI"""
        layout = QVBoxLayout(self)
        
        # Main splitter for chart builder and preview
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Chart configuration
        config_panel = QFrame()
        config_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        config_panel.setMaximumWidth(400)
        config_layout = QVBoxLayout(config_panel)
        
        # Data source selection
        data_group = QGroupBox("Data Source")
        data_layout = QFormLayout(data_group)
        
        self.data_source = QComboBox()
        self.data_source.addItems(["Select Dataset...", "Customer Analytics", "Sales Performance", "Financial Reports"])
        data_layout.addRow("Dataset:", self.data_source)
        
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Enter SQL query or select columns...")
        self.query_input.setMaximumHeight(100)
        data_layout.addRow("Query:", self.query_input)
        
        config_layout.addWidget(data_group)
        
        # Chart type selection
        chart_group = QGroupBox("Chart Configuration")
        chart_layout = QFormLayout(chart_group)
        
        self.chart_type = QComboBox()
        self.chart_type.addItems(["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram", "Box Plot"])
        chart_layout.addRow("Chart Type:", self.chart_type)
        
        self.x_axis = QComboBox()
        self.x_axis.addItems(["Select Column...", "Date", "Category", "Region", "Product"])
        chart_layout.addRow("X-Axis:", self.x_axis)
        
        self.y_axis = QComboBox()
        self.y_axis.addItems(["Select Column...", "Revenue", "Count", "Amount", "Percentage"])
        chart_layout.addRow("Y-Axis:", self.y_axis)
        
        self.group_by = QComboBox()
        self.group_by.addItems(["None", "Category", "Region", "Date"])
        chart_layout.addRow("Group By:", self.group_by)
        
        config_layout.addWidget(chart_group)
        
        # Chart styling
        style_group = QGroupBox("Styling")
        style_layout = QFormLayout(style_group)
        
        self.chart_title = QLineEdit()
        self.chart_title.setPlaceholderText("Enter chart title...")
        style_layout.addRow("Title:", self.chart_title)
        
        self.color_scheme = QComboBox()
        self.color_scheme.addItems(["Default", "Blues", "Greens", "Reds", "Rainbow"])
        style_layout.addRow("Colors:", self.color_scheme)
        
        self.show_legend = QCheckBox("Show Legend")
        self.show_legend.setChecked(True)
        style_layout.addRow(self.show_legend)
        
        config_layout.addWidget(style_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self.preview_chart)
        button_layout.addWidget(preview_btn)
        
        save_btn = QPushButton("Save Chart")
        save_btn.clicked.connect(self.save_chart)
        button_layout.addWidget(save_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_chart)
        button_layout.addWidget(export_btn)
        
        config_layout.addLayout(button_layout)
        config_layout.addStretch()
        
        main_splitter.addWidget(config_panel)
        
        # Right panel - Chart preview
        preview_panel = QFrame()
        preview_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout(preview_panel)
        
        preview_label = QLabel("Chart Preview")
        preview_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        preview_layout.addWidget(preview_label)
        
        # Placeholder for chart preview
        self.chart_preview = QLabel("Select data source and configure chart to see preview")
        self.chart_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_preview.setStyleSheet("""
            QLabel {
                border: 2px dashed #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
                color: #666;
                padding: 40px;
                font-size: 14pt;
            }
        """)
        preview_layout.addWidget(self.chart_preview)
        
        main_splitter.addWidget(preview_panel)
        main_splitter.setSizes([400, 600])
        
        layout.addWidget(main_splitter)
        
    def preview_chart(self):
        """Preview the configured chart"""
        chart_config = self.get_chart_config()
        
        # Placeholder implementation
        preview_text = f"Chart Preview:\n\n"
        preview_text += f"Type: {chart_config['type']}\n"
        preview_text += f"Data: {chart_config['data_source']}\n"
        preview_text += f"X-Axis: {chart_config['x_axis']}\n"
        preview_text += f"Y-Axis: {chart_config['y_axis']}\n"
        preview_text += f"Title: {chart_config['title']}\n\n"
        preview_text += "Chart visualization would appear here after implementing charting backend."
        
        self.chart_preview.setText(preview_text)
        self.chart_preview.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
    def save_chart(self):
        """Save chart configuration"""
        chart_config = self.get_chart_config()
        self.chart_created.emit(chart_config)
        
    def export_chart(self):
        """Export chart to file"""
        # Placeholder implementation
        logger.info("Chart export functionality to be implemented")
        
    def get_chart_config(self):
        """Get current chart configuration"""
        return {
            'type': self.chart_type.currentText(),
            'data_source': self.data_source.currentText(),
            'query': self.query_input.toPlainText(),
            'x_axis': self.x_axis.currentText(),
            'y_axis': self.y_axis.currentText(),
            'group_by': self.group_by.currentText(),
            'title': self.chart_title.text(),
            'color_scheme': self.color_scheme.currentText(),
            'show_legend': self.show_legend.isChecked()
        }

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

class AnalyticsSection(QWidget):
    """Analytics section with Charts & Reports, SQL Query, and Document Viewer"""
    
    chart_created = pyqtSignal(dict)  # chart_config
    query_executed = pyqtSignal(str, dict)  # query, results
    document_opened = pyqtSignal(str)  # document_path
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup analytics section UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Section header
        header_layout = QHBoxLayout()
        
        title = QLabel("Analytics & Reporting")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #333; padding: 8px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Quick access buttons
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setToolTip("Refresh data sources")
        refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Tab widget for different analytics tools
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
        
        # Charts & Reports tab
        self.charts_tab = ChartsReportsTab(self.config_manager)
        self.tab_widget.addTab(self.charts_tab, "📊 Charts & Reports")
        
        # SQL Query tab
        self.sql_tab = SQLQueryTab(self.config_manager)
        self.tab_widget.addTab(self.sql_tab, "🗄️ SQL Query")
        
        # Document Viewer tab
        self.document_tab = DocumentViewerTab(self.config_manager)
        self.tab_widget.addTab(self.document_tab, "📄 Document Viewer")
        
        layout.addWidget(self.tab_widget)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.charts_tab.chart_created.connect(self.chart_created.emit)
        self.sql_tab.query_executed.connect(self.query_executed.emit)
        self.document_tab.document_opened.connect(self.document_opened.emit)
        
    def refresh_data(self):
        """Refresh data sources"""
        logger.info("Refreshing analytics data sources")
        # Placeholder implementation
        # This would refresh available datasets, documents, etc.
        
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
                
    def handle_query(self, query: str, query_type: str):
        """Handle query from main search bar"""
        if query_type == "SQL Query":
            self.tab_widget.setCurrentWidget(self.sql_tab)
            self.sql_tab.query_editor.setText(query)
        elif query_type == "Document Search":
            self.tab_widget.setCurrentWidget(self.document_tab)
            # Trigger document search with query
