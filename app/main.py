#!/usr/bin/env python3
"""
Data Platform Application
Main entry point for the PyQt6 data platform application
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QStyleFactory
from PyQt6.QtCore import QDir
from PyQt6.QtGui import QIcon

from ui.main_window import MainWindow
from config.config_manager import ConfigManager
from utils.logging_setup import setup_logging

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Setup application properties
    app.setApplicationName("Data Platform")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("DataPlatform")
    app.setOrganizationDomain("dataplatform.local")
    
    # Setup logging
    setup_logging()
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Create and show main window
    main_window = MainWindow(config_manager)
    main_window.show()
    
    # Start the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()