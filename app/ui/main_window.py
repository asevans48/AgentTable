"""
Main Window for Data Platform Application
Provides the primary interface with sections, search, and file management
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, 
    QLabel, QFrame, QStatusBar, QSystemTrayIcon, QMenu,
    QTreeWidget, QTreeWidgetItem, QTabWidget, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QAction, QPixmap, QPainter, QPen, QColor, QIcon

from ui.sections.file_browser import FileBrowser
from ui.sections.dataset_browser import DatasetBrowser
from ui.widgets.search_bar import SearchBar
from ui.widgets.search_results import SearchResults
from ui.dialogs.settings_dialog import SettingsDialog
from ui.dialogs.filter_dialog import FilterDialog

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, config_manager, user_info=None):
        super().__init__()
        self.config_manager = config_manager
        self.user_info = user_info or {}
        self.current_section = None
        
        self.setWindowTitle("Data Platform - Vector Search & File Management")
        self.setMinimumSize(1200, 800)
        
        # Create and set the icon early
        self.app_icon = self.create_app_icon()
        self.setWindowIcon(self.app_icon)
        
        # Also set it at the application level
        app = QApplication.instance()
        if app:
            app.setWindowIcon(self.app_icon)
        
        self.showMaximized()
        
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()
        self.setup_system_tray()

    def show_normal(self):
        """Show and raise the main window"""
        self.show()
        self.raise_()
        self.activateWindow()
    
    def focus_search(self):
        """Show window and focus on search bar"""
        self.show_normal()
        if hasattr(self.search_bar, 'focus_search'):
            self.search_bar.focus_search()
        else:
            # Fallback: focus the search input directly
            if hasattr(self.search_bar, 'search_input'):
                self.search_bar.search_input.setFocus()
                self.search_bar.search_input.selectAll()
            
    def on_tray_activated(self, reason):
        """Handle system tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show_normal()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Single click - just show if hidden
            if not self.isVisible():
                self.show_normal()

    def exit_completely(self):
        """Exit the application completely"""
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide()
        self.close()
        
    def closeEvent(self, event):
        """Handle close event - normal close behavior"""
        # Clean up system tray icon
        if hasattr(self, 'tray_icon'):
            self.tray_icon.hide()
        
        # Accept the close event (actually close the app)
        event.accept()

    def setup_system_tray(self):
        """Setup system tray icon"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
            
        # Create a custom icon for the system tray
        self.tray_icon = QSystemTrayIcon(self)
        
        # Create custom icon
        icon = self.create_app_icon()
        self.tray_icon.setIcon(icon)
        self.setWindowIcon(icon)  # Also set as window icon
        
        # Create tray menu
        tray_menu = QMenu()
        
        # Show/Hide action
        show_action = QAction("Show Data Platform", self)
        show_action.triggered.connect(self.show_normal)
        tray_menu.addAction(show_action)
        
        hide_action = QAction("Hide to Tray", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)
        
        tray_menu.addSeparator()
        
        # Quick actions
        search_action = QAction("🔍 Quick Search", self)
        search_action.triggered.connect(self.focus_search)
        tray_menu.addAction(search_action)
        
        settings_action = QAction("⚙️ Settings", self)
        settings_action.triggered.connect(self.show_settings)
        tray_menu.addAction(settings_action)
        
        tray_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.setToolTip("Data Platform - Vector Search & File Management")
        
        # Handle tray icon activation
        self.tray_icon.activated.connect(self.on_tray_activated)
        
        # Show the tray icon
        self.tray_icon.show()

    def create_app_icon(self):
        """Create a custom application icon with multiple sizes for better Windows support"""
        icon = QIcon()
        
        # Create multiple sizes for better Windows integration
        for size in [16, 24, 32, 48, 64]:
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Scale the design based on size
            margin = max(2, size // 16)
            circle_size = size - (margin * 2)
            
            # Draw background circle
            painter.setBrush(QColor(45, 123, 251))
            painter.setPen(QPen(QColor(25, 89, 181), max(1, size // 16)))
            painter.drawEllipse(margin, margin, circle_size, circle_size)
            
            # Draw database symbol - scale with icon size
            painter.setBrush(Qt.GlobalColor.white)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            
            # Calculate proportional dimensions
            db_width = size // 2
            db_height = size // 8
            db_x = (size - db_width) // 2
            
            # Three database layers
            for i in range(3):
                y_pos = margin + (size // 6) + (i * (size // 6))
                painter.drawEllipse(db_x, y_pos, db_width, db_height)
                painter.drawRect(db_x, y_pos + db_height//2, db_width, db_height//2)
            
            painter.end()
            icon.addPixmap(pixmap)
        
        return icon
            
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
        
        # Display current user info
        username = self.user_info.get('username', 'Unknown User')
        user_label = QLabel(f"👤 {username}")
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
        self.search_results = SearchResults(self.config_manager)
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
        
        # Dataset browser tab
        self.dataset_browser = DatasetBrowser(self.config_manager)
        tab_widget.addTab(self.dataset_browser, "Datasets")

        # File browser tab
        self.file_browser = FileBrowser(self.config_manager)
        tab_widget.addTab(self.file_browser, "Files")
        
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

        logout_action = QAction("Logout", self)
        logout_action.triggered.connect(self.logout)
        file_menu.addAction(logout_action)

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
        
        tools_menu.addSeparator()
        
        rebuild_vector_action = QAction("Rebuild Vector Database", self)
        rebuild_vector_action.triggered.connect(self.rebuild_vector_database)
        tools_menu.addAction(rebuild_vector_action)
        
        tools_menu.addSeparator()
        
        cloud_auth_action = QAction("Cloud Authentication", self)
        cloud_auth_action.triggered.connect(self.show_cloud_auth)
        tools_menu.addAction(cloud_auth_action)
        
        change_password_action = QAction("Change Password", self)
        change_password_action.triggered.connect(self.show_change_password)
        tools_menu.addAction(change_password_action)
        
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
        self.search_bar.chat_triggered.connect(self.handle_chat_request)
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
        
    def handle_chat_request(self, message):
        """Handle AI chat request"""
        self.status_bar.showMessage(f"AI Chat: {message}")
        
        # For now, treat chat as a special search type
        # In the future, this would integrate with AI chat functionality
        self.search_results.perform_search(message, "AI Chat")
        
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
            # Fixed method call
            self.dataset_browser.apply_dataset_filters(filters)
            
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
        
    def rebuild_vector_database(self):
        """Rebuild the vector search database"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Rebuild Vector Database",
            "This will rebuild the entire vector search database.\n\n"
            "This may take several minutes depending on the amount of data.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Access the search results widget and trigger rebuild
            if hasattr(self.search_results, 'vector_engine'):
                try:
                    # Get watched directories for rebuilding
                    watched_dirs = self.config_manager.get("file_management.watched_directories", [])
                    if watched_dirs:
                        # Start rebuild process
                        self.status_bar.showMessage("Rebuilding vector database...")
                        
                        # Use the vector search engine to rebuild
                        from utils.vector_search import VectorSearchEngine
                        vector_engine = VectorSearchEngine(self.config_manager)
                        results = vector_engine.rebuild_index(watched_dirs)
                        
                        # Show results
                        indexed_files = results.get('indexed_files', 0)
                        total_files = results.get('total_files', 0)
                        
                        QMessageBox.information(
                            self,
                            "Rebuild Complete",
                            f"Vector database rebuilt successfully!\n\n"
                            f"Indexed {indexed_files} out of {total_files} files."
                        )
                        
                        self.status_bar.showMessage(f"Vector database rebuilt: {indexed_files}/{total_files} files indexed")
                    else:
                        QMessageBox.warning(
                            self,
                            "No Data Sources",
                            "No watched directories found.\n\n"
                            "Please add some directories in the Files tab first."
                        )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Rebuild Failed",
                        f"Failed to rebuild vector database:\n\n{str(e)}"
                    )
                    self.status_bar.showMessage("Vector database rebuild failed")
    
    def show_change_password(self):
        """Show change password dialog"""
        from ui.dialogs.login_dialog import LoginDialog
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QMessageBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Change Password")
        dialog.setModal(True)
        dialog.setFixedSize(400, 250)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel("Change your login password:")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Form for password change
        form_layout = QFormLayout()
        
        current_password = QLineEdit()
        current_password.setEchoMode(QLineEdit.EchoMode.Password)
        current_password.setPlaceholderText("Current password")
        form_layout.addRow("Current Password:", current_password)
        
        new_password = QLineEdit()
        new_password.setEchoMode(QLineEdit.EchoMode.Password)
        new_password.setPlaceholderText("New password")
        form_layout.addRow("New Password:", new_password)
        
        confirm_password = QLineEdit()
        confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        confirm_password.setPlaceholderText("Confirm new password")
        form_layout.addRow("Confirm Password:", confirm_password)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        change_button = QPushButton("Change Password")
        change_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 4px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        
        def change_password():
            if not all([current_password.text(), new_password.text(), confirm_password.text()]):
                QMessageBox.warning(dialog, "Error", "Please fill in all fields")
                return
            
            if new_password.text() != confirm_password.text():
                QMessageBox.warning(dialog, "Error", "New passwords do not match")
                return
            
            # Verify current password and update
            login_dialog = LoginDialog()
            success, message = login_dialog.create_local_account(
                self.user_info.get('username', 'admin'), 
                new_password.text()
            )
            
            if success:
                QMessageBox.information(dialog, "Success", "Password changed successfully!")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Error", f"Failed to change password: {message}")
        
        change_button.clicked.connect(change_password)
        button_layout.addWidget(change_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 4px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def show_cloud_auth(self):
        """Show cloud authentication dialog"""
        from ui.dialogs.cloud_auth_dialog import CloudAuthDialog
        
        dialog = CloudAuthDialog(self)
        dialog.credentials_updated.connect(self.on_cloud_credentials_updated)
        dialog.exec()
    
    def on_cloud_credentials_updated(self):
        """Handle cloud credentials update"""
        self.status_bar.showMessage("Cloud credentials updated successfully")
    
    def logout(self):
        """Logout and return to login screen"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self, 
            "Logout", 
            "Are you sure you want to logout?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.close()
            # Restart the application to show login again
            import sys
            import os
            os.execl(sys.executable, sys.executable, *sys.argv)
