#!/usr/bin/env python3
"""
Data Platform Application
Main entry point for the PyQt6 data platform application
"""

import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from ui.main_window import MainWindow
from ui.dialogs.login_dialog import LoginDialog
from config.config_manager import ConfigManager
from utils.logging_setup import setup_logging

def main():
    """Main application entry point"""
    # Setup logging
    setup_logging()
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Data Platform")
    app.setApplicationDisplayName("Data Platform")
    app.setApplicationVersion("1.0.0")
    
    # Windows-specific: Set App User Model ID EARLY
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('dataplatform.main.1.0')
    except:
        pass
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Show login dialog first
    login_dialog = LoginDialog()
    
    # If login is cancelled, exit the application
    if login_dialog.exec() != LoginDialog.DialogCode.Accepted:
        sys.exit(0)
    
    # Get user info from successful login
    user_info = login_dialog.get_user_info()
    
    # Create and show main window only after successful login
    main_window = MainWindow(config_manager, user_info)
    main_window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
