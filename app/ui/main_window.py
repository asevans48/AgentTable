"""
Main Window for Data Platform Application
Provides the primary interface with sections, search, and file management
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, 
    QLabel, QFrame, QStatusBar,
    QTreeWidget, QTreeWidgetItem, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QAction

from ui.sections.file_browser import FileBrowser
from ui.sections.dataset_browser import DatasetBrowser
from ui.widgets.search_bar import SearchBar
from ui.widgets.search_results import SearchResults
from ui.dialogs.settings_dialog import SettingsDialog
from ui.dialogs.filter_dialog import FilterDialog

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.current_section = None
        
        self.setWindowTitle("Data Platform - Vector Search & File Management")
        self.setMinimumSize(1200, 800)
        self.showMaximized()
        
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Sections
        self.sections_panel = self.create_sections_panel()
        main_splitter.addWidget(self.sections_panel)
        
        # Center panel - Main content area
        self.center_panel = self.create_center_panel()
        main_splitter.addWidget(self.center_panel)
        
        # Right panel - File browser and datasets
        self.right_panel = self.create_right_panel()
        main_splitter.addWidget(self.right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([250, 600, 350])
        
    def create_sections_panel(self):
        """Create the left sections panel"""
        panel = QFrame()
        panel.setMaximumWidth(300)
        panel.setMinimumWidth(200)
        panel.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-right: 1px solid #e9ecef;
                border-radius: 0px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Modern solid header with sleek design
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: none;
                border-radius: 0px;
                padding: 20px 16px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 16, 0, 16)
        header_layout.setSpacing(0)
        
        # App title with modern typography - centered
        app_title = QLabel("Data Platform")
        app_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        app_title.setStyleSheet("""
            color: #ffffff;
            margin: 0px;
            letter-spacing: 0.5px;
            text-align: center;
        """)
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(app_title)
        
        layout.addWidget(header_frame)
        
        # Sections tree with modern styling
        self.sections_tree = QTreeWidget()
        self.sections_tree.setHeaderHidden(True)
        self.sections_tree.setRootIsDecorated(False)
        self.sections_tree.setIndentation(20)
        self.sections_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #f8f9fa;
                border: none;
                outline: none;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
                padding: 8px 0px;
            }
            QTreeWidget::item {
                padding: 12px 16px;
                border: none;
                color: #495057;
                font-weight: 500;
            }
            QTreeWidget::item:hover {
                background-color: #e9ecef;
                color: #212529;
                border-radius: 0px;
            }
            QTreeWidget::item:selected {
                background-color: #007bff;
                color: white;
                border-radius: 0px;
            }
            QTreeWidget::item:selected:hover {
                background-color: #0056b3;
            }
            QTreeWidget::branch {
                background: transparent;
            }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                border-image: none;
                image: none;
            }
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {
                border-image: none;
                image: none;
            }
        """)
        
        self.populate_sections_tree()
        layout.addWidget(self.sections_tree)
        
        # Modern footer with user info
        footer_frame = QFrame()
        footer_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-top: 1px solid #e9ecef;
                padding: 12px 16px;
            }
        """)
        footer_layout = QVBoxLayout(footer_frame)
        footer_layout.setContentsMargins(16, 12, 16, 12)
        
        user_label = QLabel("👤 Current User")
        user_label.setFont(QFont("Segoe UI", 9))
        user_label.setStyleSheet("color: #6c757d; margin-bottom: 2px;")
        footer_layout.addWidget(user_label)
        
        status_label = QLabel("🟢 Connected")
        status_label.setFont(QFont("Segoe UI", 8))
        status_label.setStyleSheet("color: #28a745;")
        footer_layout.addWidget(status_label)
        
        layout.addWidget(footer_frame)
        
        return panel
        
    def create_center_panel(self):
        """Create the main center content panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Prominent search bar
        self.search_bar = SearchBar(self.config_manager)
        layout.addWidget(self.search_bar)
        
        # Search results area
        self.search_results = SearchResults()
        layout.addWidget(self.search_results)
        
        return panel
        
    def create_right_panel(self):
        """Create the right panel with file browser and datasets"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(400)
        panel.setMinimumWidth(300)
        
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        tab_widget = QTabWidget()
        
        # File browser tab
        self.file_browser = FileBrowser(self.config_manager)
        tab_widget.addTab(self.file_browser, "Files")
        
        # Dataset browser tab
        self.dataset_browser = DatasetBrowser(self.config_manager)
        tab_widget.addTab(self.dataset_browser, "Datasets")
        
        layout.addWidget(tab_widget)
        
        return panel
        
    def populate_sections_tree(self):
        """Populate the sections tree with available sections"""
        sections_data = [
            ("🔍 Search", ["Vector Search"]),
            ("📊 Data Management", ["Table", "Data Quality"]),
            ("📈 Analytics", ["Charts & Reports", "SQL Query", "Document Viewer"]),
            ("⚡ Workflows", ["Agent Builder", "Task Manager", "Scheduler"]),
            ("🛡️ Governance", ["Documentation", "Permissions", "Metadata"]),
            ("🚀 Applications", ["App Builder", "Configuration", "Export"])
        ]
        
        for section_name, subsections in sections_data:
            # Create main section item
            section_item = QTreeWidgetItem([section_name])
            section_item.setFont(0, QFont("Segoe UI", 10, QFont.Weight.Bold))
            section_item.setExpanded(True)
            
            # Style the main section
            section_item.setData(0, Qt.ItemDataRole.UserRole, "section")
            
            for subsection in subsections:
                subsection_item = QTreeWidgetItem([f"  {subsection}"])
                subsection_item.setFont(0, QFont("Segoe UI", 9))
                subsection_item.setData(0, Qt.ItemDataRole.UserRole, "subsection")
                section_item.addChild(subsection_item)
                
            self.sections_tree.addTopLevelItem(section_item)
        
        # Expand all sections by default
        self.sections_tree.expandAll()
            
    def setup_menu_bar(self):
        """Setup the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Application", self)
        new_action.triggered.connect(self.new_application)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Application", self)
        open_action.triggered.connect(self.open_application)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Application", self)
        save_action.triggered.connect(self.save_application)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        filter_action = QAction("Filter Datasets", self)
        filter_action.triggered.connect(self.show_filter_dialog)
        tools_menu.addAction(filter_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_data)
        view_menu.addAction(refresh_action)
        
    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Welcome to Data Platform")
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.sections_tree.itemClicked.connect(self.on_section_clicked)
        self.search_bar.search_triggered.connect(self.perform_search)
        self.file_browser.file_selected.connect(self.on_file_selected)
        self.dataset_browser.dataset_selected.connect(self.on_dataset_selected)
        
    def on_section_clicked(self, item, column):
        """Handle section tree item clicks"""
        section_text = item.text(0)
        self.status_bar.showMessage(f"Selected: {section_text}")
        
        # Here you would switch to the appropriate section/tab
        # For now, just update the status
        
    def perform_search(self, query, search_type):
        """Perform search based on query and type"""
        self.status_bar.showMessage(f"Searching for: {query}")
        
        # Trigger search in search results widget
        self.search_results.perform_search(query, search_type)
        
    def on_file_selected(self, file_path):
        """Handle file selection from file browser"""
        self.status_bar.showMessage(f"Selected file: {file_path}")
        
    def on_dataset_selected(self, dataset_info):
        """Handle dataset selection from dataset browser"""
        self.status_bar.showMessage(f"Selected dataset: {dataset_info.get('name', 'Unknown')}")
        
    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.config_manager, self)
        dialog.exec()
        
    def show_filter_dialog(self):
        """Show filter dialog for datasets"""
        dialog = FilterDialog(self)
        if dialog.exec():
            filters = dialog.get_filters()
            self.dataset_browser.apply_filters(filters)
            
    def new_application(self):
        """Create new application configuration"""
        self.config_manager.new_application()
        self.status_bar.showMessage("New application created")
        
    def open_application(self):
        """Open existing application configuration"""
        # Would show file dialog to select application config
        self.status_bar.showMessage("Open application - TODO: Implement file dialog")
        
    def save_application(self):
        """Save current application configuration"""
        self.config_manager.save_application()
        self.status_bar.showMessage("Application saved")
        
    def refresh_data(self):
        """Refresh all data views"""
        self.file_browser.refresh()
        self.dataset_browser.refresh()
        self.status_bar.showMessage("Data refreshed")
