"""
Search Bar Widget
Prominent search interface for vector search and AI chat
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, 
    QComboBox, QLabel, QFrame, QToolButton, QMenu
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut, QAction
import logging

logger = logging.getLogger(__name__)

class SearchBar(QWidget):
    """Prominent search bar widget for vector search and AI interactions"""
    
    # Signals
    search_triggered = pyqtSignal(str, str)  # query, search_type
    chat_triggered = pyqtSignal(str)  # message
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.search_history = []
        self.current_history_index = -1
        
        self.setup_ui()
        self.setup_connections()
        self.load_recent_searches()
        
    def setup_ui(self):
        """Setup the search bar UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Main search container
        search_container = QFrame()
        search_container.setFrameStyle(QFrame.Shape.StyledPanel)
        search_container.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 8px;
            }
            QFrame:focus-within {
                border-color: #007bff;
            }
        """)
        
        search_layout = QHBoxLayout(search_container)
        search_layout.setContentsMargins(8, 8, 8, 8)
        
        # Search type selector
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems([
            "Vector Search",
            "AI Chat",
            "SQL Query",
            "Document Search",
            "Dataset Search"
        ])
        self.search_type_combo.setMinimumWidth(120)
        self.search_type_combo.setToolTip("Select search type")
        search_layout.addWidget(self.search_type_combo)
        
        # Main search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search documents, chat with data, or enter SQL queries...")
        self.search_input.setMinimumHeight(40)
        self.search_input.setFont(QFont("Arial", 11))
        self.search_input.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                font-size: 11pt;
                padding: 8px;
            }
        """)
        search_layout.addWidget(self.search_input, 1)
        
        # Advanced options button
        self.options_button = QToolButton()
        self.options_button.setText("⚙️")
        self.options_button.setToolTip("Advanced options")
        self.options_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setup_options_menu()
        search_layout.addWidget(self.options_button)
        
        # Search button
        self.search_button = QPushButton("Search")
        self.search_button.setMinimumHeight(40)
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        search_layout.addWidget(self.search_button)
        
        layout.addWidget(search_container)
        
        # Quick filters/suggestions bar
        self.setup_quick_filters()
        layout.addWidget(self.quick_filters_container)
        
    def setup_quick_filters(self):
        """Setup quick filter buttons"""
        self.quick_filters_container = QFrame()
        filters_layout = QHBoxLayout(self.quick_filters_container)
        filters_layout.setContentsMargins(0, 5, 0, 0)
        
        # Recent searches label
        recent_label = QLabel("Recent:")
        recent_label.setFont(QFont("Arial", 9))
        recent_label.setStyleSheet("color: #6c757d;")
        filters_layout.addWidget(recent_label)
        
        # Quick filter buttons (will be populated dynamically)
        self.quick_filter_buttons = []
        
        filters_layout.addStretch()
        
        # Clear history button
        clear_button = QPushButton("Clear History")
        clear_button.setFont(QFont("Arial", 8))
        clear_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                padding: 2px 8px;
                color: #6c757d;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
            }
        """)
        clear_button.clicked.connect(self.clear_search_history)
        filters_layout.addWidget(clear_button)
        
    def setup_options_menu(self):
        """Setup the advanced options menu"""
        menu = QMenu(self)
        
        # Search scope options
        scope_menu = menu.addMenu("Search Scope")
        
        all_datasets_action = QAction("All Datasets", self)
        all_datasets_action.setCheckable(True)
        all_datasets_action.setChecked(True)
        scope_menu.addAction(all_datasets_action)
        
        accessible_only_action = QAction("Accessible Only", self)
        accessible_only_action.setCheckable(True)
        scope_menu.addAction(accessible_only_action)
        
        recent_files_action = QAction("Recent Files", self)
        recent_files_action.setCheckable(True)
        scope_menu.addAction(recent_files_action)
        
        menu.addSeparator()
        
        # AI model selection (when chat mode is active)
        ai_menu = menu.addMenu("AI Model")
        
        claude_action = QAction("Claude (Anthropic)", self)
        claude_action.setCheckable(True)
        claude_action.setChecked(self.config_manager.is_ai_tool_enabled("anthropic"))
        claude_action.setEnabled(self.config_manager.is_ai_tool_enabled("anthropic"))
        ai_menu.addAction(claude_action)
        
        gpt_action = QAction("GPT (OpenAI)", self)
        gpt_action.setCheckable(True)
        gpt_action.setEnabled(self.config_manager.is_ai_tool_enabled("openai"))
        ai_menu.addAction(gpt_action)
        
        local_action = QAction("Local Model", self)
        local_action.setCheckable(True)
        local_action.setEnabled(self.config_manager.is_ai_tool_enabled("local_models"))
        ai_menu.addAction(local_action)
        
        menu.addSeparator()
        
        # Search options
        case_sensitive_action = QAction("Case Sensitive", self)
        case_sensitive_action.setCheckable(True)
        menu.addAction(case_sensitive_action)
        
        regex_action = QAction("Regular Expression", self)
        regex_action.setCheckable(True)
        menu.addAction(regex_action)
        
        self.options_button.setMenu(menu)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.search_button.clicked.connect(self.perform_search)
        self.search_input.returnPressed.connect(self.perform_search)
        self.search_type_combo.currentTextChanged.connect(self.on_search_type_changed)
        
        
        # Setup keyboard shortcuts
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        search_shortcut.activated.connect(self.focus_search)
        
        # History navigation
        up_shortcut = QShortcut(QKeySequence("Up"), self.search_input)
        up_shortcut.activated.connect(self.navigate_history_up)
        
        down_shortcut = QShortcut(QKeySequence("Down"), self.search_input)
        down_shortcut.activated.connect(self.navigate_history_down)
        
    def on_search_type_changed(self, search_type):
        """Handle search type changes"""
        if search_type == "AI Chat":
            self.search_input.setPlaceholderText("Ask questions about your data...")
            self.search_button.setText("Send")
        elif search_type == "SQL Query":
            self.search_input.setPlaceholderText("Enter SQL query...")
            self.search_button.setText("Execute")
        elif search_type == "Vector Search":
            self.search_input.setPlaceholderText("Search documents and data...")
            self.search_button.setText("Search")
        else:
            self.search_input.setPlaceholderText("Search...")
            self.search_button.setText("Search")
            
    def perform_search(self):
        """Perform search based on current input and type"""
        query = self.search_input.text().strip()
        if not query:
            return
            
        search_type = self.search_type_combo.currentText()
        
        # Add to search history
        self.add_to_history(query)
        
        # Emit appropriate signal
        if search_type == "AI Chat":
            self.chat_triggered.emit(query)
        else:
            self.search_triggered.emit(query, search_type)
            
        logger.info(f"Performing {search_type}: {query}")
        
    def add_to_history(self, query):
        """Add query to search history"""
        if query in self.search_history:
            self.search_history.remove(query)
        
        self.search_history.insert(0, query)
        
        # Limit history size
        if len(self.search_history) > 10:
            self.search_history = self.search_history[:10]
            
        self.current_history_index = -1
        self.update_quick_filters()
        self.save_recent_searches()
        
    def navigate_history_up(self):
        """Navigate up in search history"""
        if not self.search_history:
            return
            
        if self.current_history_index < len(self.search_history) - 1:
            self.current_history_index += 1
            self.search_input.setText(self.search_history[self.current_history_index])
            
    def navigate_history_down(self):
        """Navigate down in search history"""
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self.search_input.setText(self.search_history[self.current_history_index])
        elif self.current_history_index == 0:
            self.current_history_index = -1
            self.search_input.clear()
            
    def update_quick_filters(self):
        """Update quick filter buttons with recent searches"""
        # Clear existing buttons
        for button in self.quick_filter_buttons:
            button.deleteLater()
        self.quick_filter_buttons.clear()
        
        # Add recent searches as quick filters
        layout = self.quick_filters_container.layout()
        for i, query in enumerate(self.search_history[:5]):  # Show max 5 recent
            button = QPushButton(query[:20] + "..." if len(query) > 20 else query)
            button.setFont(QFont("Arial", 8))
            button.setStyleSheet("""
                QPushButton {
                    background-color: #e9ecef;
                    border: 1px solid #ced4da;
                    border-radius: 12px;
                    padding: 2px 8px;
                    margin: 1px;
                }
                QPushButton:hover {
                    background-color: #dee2e6;
                }
            """)
            button.clicked.connect(lambda checked, q=query: self.use_quick_filter(q))
            
            # Insert before the stretch and clear button
            layout.insertWidget(layout.count() - 2, button)
            self.quick_filter_buttons.append(button)
            
    def use_quick_filter(self, query):
        """Use a quick filter query"""
        self.search_input.setText(query)
        self.perform_search()
        
    def clear_search_history(self):
        """Clear search history"""
        self.search_history.clear()
        self.current_history_index = -1
        self.update_quick_filters()
        self.save_recent_searches()
        
    def focus_search(self):
        """Focus the search input"""
        self.search_input.setFocus()
        self.search_input.selectAll()
        
    def load_recent_searches(self):
        """Load recent searches from config"""
        recent = self.config_manager.get("ui_preferences.recent_searches", [])
        self.search_history = recent[:10]  # Limit to 10
        self.update_quick_filters()
        
    def save_recent_searches(self):
        """Save recent searches to config"""
        self.config_manager.set("ui_preferences.recent_searches", self.search_history)
        
        
    def get_current_query(self):
        """Get the current search query"""
        return self.search_input.text().strip()
        
    def set_query(self, query):
        """Set the search query"""
        self.search_input.setText(query)
        
    def get_search_type(self):
        """Get the current search type"""
        return self.search_type_combo.currentText()
