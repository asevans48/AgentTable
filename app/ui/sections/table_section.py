"""
Table Section Widget
AirTable-like interface for data viewing, editing, and management
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QComboBox, QTabWidget, QFrame,
    QMenu, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QAction
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TableImportWorker(QThread):
    """Background worker for importing data into tables"""
    
    import_progress = pyqtSignal(int, str)  # progress, status
    import_complete = pyqtSignal(dict)  # import_result
    import_error = pyqtSignal(str)  # error_message
    
    def __init__(self, file_paths: List[str], import_config: Dict[str, Any]):
        super().__init__()
        self.file_paths = file_paths
        self.import_config = import_config
        
    def run(self):
        """Run the import process"""
        try:
            self.import_progress.emit(10, "Starting import...")
            
            # Placeholder implementation for file import
            total_files = len(self.file_paths)
            imported_records = 0
            
            for i, file_path in enumerate(self.file_paths):
                self.import_progress.emit(
                    int((i / total_files) * 80) + 10, 
                    f"Processing {file_path}..."
                )
                
                # Simulate processing time
                import time
                time.sleep(0.5)
                
                # Mock record count
                imported_records += 100
                
            self.import_progress.emit(100, "Import complete")
            
            result = {
                'status': 'success',
                'files_processed': total_files,
                'records_imported': imported_records,
                'table_name': self.import_config.get('table_name', 'Imported Data')
            }
            
            self.import_complete.emit(result)
            
        except Exception as e:
            logger.error(f"Import error: {e}")
            self.import_error.emit(str(e))

class DataTableWidget(QTableWidget):
    """Enhanced table widget with data management features"""
    
    cell_changed = pyqtSignal(int, int, str)  # row, column, new_value
    row_added = pyqtSignal(dict)  # row_data
    row_deleted = pyqtSignal(int)  # row_index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_table()
        self.setup_context_menu()
        
    def setup_table(self):
        """Setup table properties"""
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: white;
                alternate-background-color: #f8f9fa;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
            }
            QHeaderView::section {
                background-color: #f1f3f4;
                padding: 8px;
                border: 1px solid #e0e0e0;
                font-weight: bold;
            }
        """)
        
        # Enable editing
        self.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | 
                           QTableWidget.EditTrigger.EditKeyPressed)
        
        # Connect signals
        self.cellChanged.connect(self.on_cell_changed)
        
    def setup_context_menu(self):
        """Setup right-click context menu"""
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
    def show_context_menu(self, position):
        """Show context menu"""
        menu = QMenu(self)
        
        # Add row
        add_row_action = QAction("Add Row", self)
        add_row_action.triggered.connect(self.add_row)
        menu.addAction(add_row_action)
        
        # Delete row
        if self.currentRow() >= 0:
            delete_row_action = QAction("Delete Row", self)
            delete_row_action.triggered.connect(self.delete_current_row)
            menu.addAction(delete_row_action)
            
        menu.addSeparator()
        
        # Copy/Paste
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_selection)
        menu.addAction(copy_action)
        
        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self.paste_selection)
        menu.addAction(paste_action)
        
        menu.exec(self.mapToGlobal(position))
        
    def on_cell_changed(self, row, column):
        """Handle cell value changes"""
        new_value = self.item(row, column).text() if self.item(row, column) else ""
        self.cell_changed.emit(row, column, new_value)
        
    def add_row(self):
        """Add new row to table"""
        row_count = self.rowCount()
        self.insertRow(row_count)
        
        # Initialize cells with empty values
        for col in range(self.columnCount()):
            self.setItem(row_count, col, QTableWidgetItem(""))
            
        row_data = {}
        for col in range(self.columnCount()):
            header = self.horizontalHeaderItem(col)
            column_name = header.text() if header else f"Column {col}"
            row_data[column_name] = ""
            
        self.row_added.emit(row_data)
        
    def delete_current_row(self):
        """Delete currently selected row"""
        current_row = self.currentRow()
        if current_row >= 0:
            reply = QMessageBox.question(
                self, 
                "Delete Row", 
                f"Are you sure you want to delete row {current_row + 1}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.removeRow(current_row)
                self.row_deleted.emit(current_row)
                
    def copy_selection(self):
        """Copy selected cells to clipboard"""
        # Placeholder implementation
        logger.info("Copy operation - to be implemented")
        
    def paste_selection(self):
        """Paste from clipboard to selected cells"""
        # Placeholder implementation
        logger.info("Paste operation - to be implemented")
        
    def load_data(self, data: List[Dict[str, Any]]):
        """Load data into the table"""
        if not data:
            return
            
        # Set up columns based on first row
        first_row = data[0]
        columns = list(first_row.keys())
        
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        
        # Set up rows
        self.setRowCount(len(data))
        
        # Populate data
        for row_idx, row_data in enumerate(data):
            for col_idx, column in enumerate(columns):
                value = str(row_data.get(column, ""))
                self.setItem(row_idx, col_idx, QTableWidgetItem(value))
                
        # Auto-resize columns
        self.resizeColumnsToContents()
        
    def get_data(self) -> List[Dict[str, Any]]:
        """Get all data from the table"""
        data = []
        
        # Get column names
        columns = []
        for col in range(self.columnCount()):
            header = self.horizontalHeaderItem(col)
            columns.append(header.text() if header else f"Column {col}")
            
        # Get row data
        for row in range(self.rowCount()):
            row_data = {}
            for col, column_name in enumerate(columns):
                item = self.item(row, col)
                row_data[column_name] = item.text() if item else ""
            data.append(row_data)
            
        return data

class TableViewTab(QWidget):
    """Main table view tab for displaying and editing data"""
    
    data_changed = pyqtSignal(list)  # updated_data
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.current_table_data = []
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup table view UI"""
        layout = QVBoxLayout(self)
        
        # Table controls
        controls_layout = QHBoxLayout()
        
        # Table selection
        controls_layout.addWidget(QLabel("Table:"))
        self.table_selector = QComboBox()
        self.table_selector.addItems(["Customer Data", "Sales Records", "Inventory", "New Table..."])
        controls_layout.addWidget(self.table_selector)
        
        controls_layout.addStretch()
        
        # Action buttons
        add_row_btn = QPushButton("+ Add Row")
        add_row_btn.clicked.connect(self.add_new_row)
        controls_layout.addWidget(add_row_btn)
        
        import_btn = QPushButton("📂 Import")
        import_btn.clicked.connect(self.import_data)
        controls_layout.addWidget(import_btn)
        
        export_btn = QPushButton("💾 Export")
        export_btn.clicked.connect(self.export_data)
        controls_layout.addWidget(export_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.setStyleSheet("""
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
        save_btn.clicked.connect(self.save_table)
        controls_layout.addWidget(save_btn)
        
        layout.addLayout(controls_layout)
        
        # Data table
        self.data_table = DataTableWidget()
        layout.addWidget(self.data_table)
        
        # Load sample data
        self.load_sample_data()
        
    def setup_connections(self):
        """Setup signal connections"""
        self.data_table.cell_changed.connect(self.on_data_changed)
        self.data_table.row_added.connect(self.on_row_added)
        self.data_table.row_deleted.connect(self.on_row_deleted)
        self.table_selector.currentTextChanged.connect(self.on_table_changed)
        
    def load_sample_data(self):
        """Load sample data for demonstration"""
        sample_data = [
            {"ID": "1", "Name": "John Smith", "Email": "john@example.com", "Department": "Sales", "Salary": "65000"},
            {"ID": "2", "Name": "Jane Doe", "Email": "jane@example.com", "Department": "Marketing", "Salary": "70000"},
            {"ID": "3", "Name": "Bob Johnson", "Email": "bob@example.com", "Department": "Engineering", "Salary": "85000"},
            {"ID": "4", "Name": "Alice Wilson", "Email": "alice@example.com", "Department": "HR", "Salary": "60000"},
            {"ID": "5", "Name": "Charlie Brown", "Email": "charlie@example.com", "Department": "Sales", "Salary": "68000"}
        ]
        
        self.current_table_data = sample_data
        self.data_table.load_data(sample_data)
        
    def on_data_changed(self, row, column, new_value):
        """Handle data changes"""
        if row < len(self.current_table_data):
            columns = list(self.current_table_data[0].keys())
            if column < len(columns):
                column_name = columns[column]
                self.current_table_data[row][column_name] = new_value
                self.data_changed.emit(self.current_table_data)
                
    def on_row_added(self, row_data):
        """Handle new row addition"""
        self.current_table_data.append(row_data)
        self.data_changed.emit(self.current_table_data)
        
    def on_row_deleted(self, row_index):
        """Handle row deletion"""
        if row_index < len(self.current_table_data):
            del self.current_table_data[row_index]
            self.data_changed.emit(self.current_table_data)
            
    def on_table_changed(self, table_name):
        """Handle table selection change"""
        if table_name == "New Table...":
            self.create_new_table()
        else:
            # Load data for selected table
            logger.info(f"Loading table: {table_name}")
            
    def add_new_row(self):
        """Add new row to the table"""
        self.data_table.add_row()
        
    def import_data(self):
        """Import data from file"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files to import",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_paths:
            # For now, just show a message
            QMessageBox.information(
                self,
                "Import Data",
                f"Selected {len(file_paths)} file(s) for import. Import functionality will be implemented with the backend."
            )
            
    def export_data(self):
        """Export table data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export table data",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
        )
        
        if file_path:
            QMessageBox.information(
                self,
                "Export Data",
                f"Table data would be exported to {file_path}. Export functionality will be implemented with the backend."
            )
            
    def save_table(self):
        """Save current table state"""
        # Get current data from table
        current_data = self.data_table.get_data()
        self.current_table_data = current_data
        
        # Save to configuration
        table_name = self.table_selector.currentText()
        # Implementation would save to config manager
        
        QMessageBox.information(self, "Save", f"Table '{table_name}' saved successfully!")
        
    def create_new_table(self):
        """Create a new table"""
        # For now, just reset the table selector
        self.table_selector.setCurrentIndex(0)
        
        # Clear table and add basic structure
        self.data_table.clear()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["Column 1", "Column 2", "Column 3"])
        self.data_table.setRowCount(1)
        
        for col in range(3):
            self.data_table.setItem(0, col, QTableWidgetItem(""))

class TableFormsTab(QWidget):
    """Forms creation and management tab"""
    
    form_created = pyqtSignal(dict)  # form_config
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        
    def setup_ui(self):
        """Setup forms UI"""
        layout = QVBoxLayout(self)
        
        # Form builder placeholder
        placeholder_label = QLabel("📝 Form Builder")
        placeholder_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder_label)
        
        description = QLabel("Create custom forms to populate tables. Forms can be built manually, with AI assistance, or both.")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        description.setStyleSheet("color: #666; margin: 20px; font-size: 12pt;")
        layout.addWidget(description)
        
        # Placeholder content
        features_frame = QFrame()
        features_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 20px;
                margin: 20px;
            }
        """)
        
        features_layout = QVBoxLayout(features_frame)
        
        features = [
            "🎨 Visual form designer",
            "🤖 AI-powered form generation",
            "📊 Integration with SharePoint and MS Forms",
            "⚡ Real-time form preview",
            "🔗 Direct table population",
            "📱 Mobile-responsive forms"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("color: #555; margin: 5px 0; font-size: 11pt;")
            features_layout.addWidget(feature_label)
            
        layout.addWidget(features_frame)
        
        layout.addStretch()

class TableSection(QWidget):
    """Table section - AirTable-like interface for data management"""
    
    data_updated = pyqtSignal(str, list)  # table_name, data
    form_created = pyqtSignal(dict)  # form_config
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup table section UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Section header
        header_layout = QHBoxLayout()
        
        title = QLabel("Table - Data Management")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #333; padding: 8px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Quick access buttons
        new_table_btn = QPushButton("+ New Table")
        new_table_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        new_table_btn.clicked.connect(self.create_new_table)
        header_layout.addWidget(new_table_btn)
        
        layout.addLayout(header_layout)
        
        # Tab widget for table functionality
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
        
        # Table View tab
        self.table_tab = TableViewTab(self.config_manager)
        self.tab_widget.addTab(self.table_tab, "📊 Table View")
        
        # Forms tab
        self.forms_tab = TableFormsTab(self.config_manager)
        self.tab_widget.addTab(self.forms_tab, "📝 Forms")
        
        layout.addWidget(self.tab_widget)
        
    def setup_connections(self):
        """Setup signal connections"""
        self.table_tab.data_changed.connect(lambda data: self.data_updated.emit("current_table", data))
        self.forms_tab.form_created.connect(self.form_created.emit)
        
    def create_new_table(self):
        """Create a new table"""
        logger.info("Creating new table")
        # Switch to table view and create new table
        self.tab_widget.setCurrentWidget(self.table_tab)
        self.table_tab.create_new_table()
        
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
