#!/usr/bin/env python3
"""
Data Platform Application
Main entry point for the PyQt6 data platform application
"""

import sys

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
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
    
    # Create config manager and main window
    config_manager = ConfigManager()
    main_window = MainWindow(config_manager)
    
    # Show the main window
    main_window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()