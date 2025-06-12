"""
Search Bar Widget
Prominent search interface for vector search and AI chat
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, 
    QComboBox, QLabel, QFrame, QToolButton, QMenu, QCompleter, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import pyqtSignal, QStringListModel, Qt, QTimer
from PyQt6.QtGui import QFont, QKeySequence, QShortcut, QAction
import logging
from pathlib import Path

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
        
        # Autocomplete data
        self.dataset_names = []
        self.document_names = []
        self.file_extensions = []
        self.autocomplete_timer = QTimer()
        self.autocomplete_timer.setSingleShot(True)
        self.autocomplete_timer.timeout.connect(self.update_autocomplete_data)
        
        self.setup_ui()
        self.setup_connections()
        self.load_recent_searches()
        self.setup_autocomplete()
        
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
            "Document Search", 
            "Dataset Search",
            "AI Chat",
            "SQL Query"
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
        
        # Autocomplete connections
        self.search_input.textChanged.connect(self.on_text_changed)
        
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
            self.search_input.setPlaceholderText("Ask questions about selected items...")
            self.search_button.setText("Chat")
        elif search_type == "SQL Query":
            self.search_input.setPlaceholderText("SELECT * FROM table_name WHERE...")
            self.search_button.setText("Execute")
        elif search_type == "Vector Search":
            self.search_input.setPlaceholderText("Search for similar content using AI...")
            self.search_button.setText("Search")
        elif search_type == "Document Search":
            self.search_input.setPlaceholderText("Search document names and files...")
            self.search_button.setText("Search")
        elif search_type == "Dataset Search":
            self.search_input.setPlaceholderText("Search dataset names and descriptions...")
            self.search_button.setText("Search")
        else:
            self.search_input.setPlaceholderText("Search...")
            self.search_button.setText("Search")
            
        # Update autocomplete based on search type
        self.update_autocomplete_for_search_type(search_type)
            
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
    
    def setup_autocomplete(self):
        """Setup autocomplete functionality"""
        # Create completer
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        
        # Set initial model
        self.completer_model = QStringListModel()
        self.completer.setModel(self.completer_model)
        self.search_input.setCompleter(self.completer)
        
        # Load initial autocomplete data
        self.load_autocomplete_data()
    
    def load_autocomplete_data(self):
        """Load autocomplete data from various sources"""
        try:
            # Load dataset names
            datasets = self.config_manager.get_registered_datasets()
            self.dataset_names = [dataset['name'] for dataset in datasets if 'name' in dataset]
            
            # Load document names from watched directories
            self.document_names = []
            self.file_extensions = set()
            
            watched_dirs = self.config_manager.get("file_management.watched_directories", [])
            for directory in watched_dirs:
                if Path(directory).exists():
                    try:
                        # Get file names from directory (limit to prevent performance issues)
                        file_count = 0
                        for file_path in Path(directory).rglob("*"):
                            if file_path.is_file() and file_count < 1000:  # Limit for performance
                                self.document_names.append(file_path.name)
                                if file_path.suffix:
                                    self.file_extensions.add(file_path.suffix.lower())
                                file_count += 1
                            elif file_count >= 1000:
                                break
                    except Exception as e:
                        logger.warning(f"Error reading directory {directory}: {e}")
            
            # Convert extensions to searchable format
            self.file_extensions = [ext.lstrip('.') for ext in self.file_extensions if ext]
            
            # Update autocomplete for current search type
            current_type = self.search_type_combo.currentText()
            self.update_autocomplete_for_search_type(current_type)
            
        except Exception as e:
            logger.error(f"Error loading autocomplete data: {e}")
    
    def update_autocomplete_for_search_type(self, search_type):
        """Update autocomplete suggestions based on search type"""
        suggestions = []
        
        if search_type == "Dataset Search":
            # Add dataset names
            suggestions.extend(self.dataset_names)
            
            # Add common dataset search terms
            suggestions.extend([
                "table:", "view:", "csv:", "json:", "database:",
                "owner:", "type:", "tags:", "description:"
            ])
            
        elif search_type == "Document Search":
            # Add document names (limit to prevent overwhelming)
            suggestions.extend(self.document_names[:500])
            
            # Add file extensions
            suggestions.extend([f"*.{ext}" for ext in self.file_extensions])
            
            # Add common document search terms
            suggestions.extend([
                "*.pdf", "*.docx", "*.txt", "*.csv", "*.xlsx", "*.json", "*.xml",
                "*.py", "*.sql", "*.md", "*.html", "*.js", "*.css"
            ])
            
        elif search_type == "Vector Search":
            # Add semantic search suggestions
            suggestions.extend([
                "find documents about", "similar to", "related to",
                "documents containing", "files with", "data about"
            ])
            
        elif search_type == "SQL Query":
            # Add SQL keywords and common patterns
            suggestions.extend([
                "SELECT * FROM", "SELECT COUNT(*) FROM", "SELECT DISTINCT",
                "WHERE", "ORDER BY", "GROUP BY", "HAVING", "JOIN", "INNER JOIN",
                "LEFT JOIN", "RIGHT JOIN", "UNION", "INSERT INTO", "UPDATE",
                "DELETE FROM", "CREATE TABLE", "ALTER TABLE", "DROP TABLE"
            ])
            
        elif search_type == "AI Chat":
            # Add common AI chat starters
            suggestions.extend([
                "What is", "How does", "Can you explain", "Show me", "Find",
                "Summarize", "Compare", "Analyze", "What are the differences",
                "Tell me about", "List all", "Count the", "Calculate"
            ])
        
        # Add search history to suggestions
        suggestions.extend(self.search_history)
        
        # Remove duplicates and sort
        unique_suggestions = list(set(suggestions))
        unique_suggestions.sort()
        
        # Update completer model
        self.completer_model.setStringList(unique_suggestions)
    
    def on_text_changed(self, text):
        """Handle text changes for dynamic autocomplete"""
        # Trigger autocomplete data refresh after a delay
        self.autocomplete_timer.start(2000)  # 2 second delay
        
        # For dataset search, provide dynamic suggestions based on current text
        if self.search_type_combo.currentText() == "Dataset Search" and len(text) >= 2:
            self.provide_dynamic_dataset_suggestions(text)
        elif self.search_type_combo.currentText() == "Document Search" and len(text) >= 2:
            self.provide_dynamic_document_suggestions(text)
    
    def provide_dynamic_dataset_suggestions(self, text):
        """Provide dynamic suggestions for dataset search"""
        try:
            text_lower = text.lower()
            dynamic_suggestions = []
            
            # Search through dataset metadata
            datasets = self.config_manager.get_registered_datasets()
            for dataset in datasets:
                # Check name
                if text_lower in dataset.get('name', '').lower():
                    dynamic_suggestions.append(dataset['name'])
                
                # Check description
                description = dataset.get('description', '')
                if text_lower in description.lower():
                    # Add relevant words from description
                    words = description.split()
                    for word in words:
                        if len(word) > 3 and text_lower in word.lower():
                            dynamic_suggestions.append(word)
                
                # Check tags
                for tag in dataset.get('tags', []):
                    if text_lower in tag.lower():
                        dynamic_suggestions.append(tag)
                
                # Check owner
                owner = dataset.get('owner', '')
                if text_lower in owner.lower():
                    dynamic_suggestions.append(f"owner:{owner}")
            
            # Update suggestions if we found any
            if dynamic_suggestions:
                current_suggestions = self.completer_model.stringList()
                combined_suggestions = list(set(current_suggestions + dynamic_suggestions))
                combined_suggestions.sort()
                self.completer_model.setStringList(combined_suggestions)
                
        except Exception as e:
            logger.error(f"Error providing dynamic dataset suggestions: {e}")
    
    def provide_dynamic_document_suggestions(self, text):
        """Provide dynamic suggestions for document search"""
        try:
            text_lower = text.lower()
            dynamic_suggestions = []
            
            # If text looks like a file extension search
            if text.startswith('*.') or text.startswith('.'):
                ext = text.lstrip('*.')
                matching_extensions = [f"*.{e}" for e in self.file_extensions if e.startswith(ext)]
                dynamic_suggestions.extend(matching_extensions)
            
            # Search through document names
            for doc_name in self.document_names:
                if text_lower in doc_name.lower():
                    dynamic_suggestions.append(doc_name)
                    
                    # Also suggest the directory/path context
                    if '.' in doc_name:
                        name_without_ext = doc_name.rsplit('.', 1)[0]
                        if text_lower in name_without_ext.lower():
                            dynamic_suggestions.append(name_without_ext)
            
            # Limit suggestions to prevent performance issues
            dynamic_suggestions = dynamic_suggestions[:100]
            
            # Update suggestions if we found any
            if dynamic_suggestions:
                current_suggestions = self.completer_model.stringList()
                combined_suggestions = list(set(current_suggestions + dynamic_suggestions))
                combined_suggestions.sort()
                self.completer_model.setStringList(combined_suggestions)
                
        except Exception as e:
            logger.error(f"Error providing dynamic document suggestions: {e}")
    
    def update_autocomplete_data(self):
        """Update autocomplete data (called by timer)"""
        self.load_autocomplete_data()
        
    def get_current_query(self):
        """Get the current search query"""
        return self.search_input.text().strip()
        
    def set_query(self, query):
        """Set the search query"""
        self.search_input.setText(query)
        
    def get_search_type(self):
        """Get the current search type"""
        return self.search_type_combo.currentText()
