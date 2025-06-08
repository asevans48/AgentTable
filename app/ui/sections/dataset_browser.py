"""
Dataset Browser Widget
Shows registered datasets with filtering and access information
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QLineEdit, QFrame, QScrollArea,
    QMessageBox, QDialog, QTextEdit, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatasetItem(QFrame):
    """Individual dataset item widget"""
    
    dataset_selected = pyqtSignal(dict)  # dataset_info
    chat_requested = pyqtSignal(dict)  # dataset_info
    access_requested = pyqtSignal(dict)  # dataset_info
    
    def __init__(self, dataset_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.dataset_info = dataset_info
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dataset item UI"""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin: 1px;
                padding: 8px;
            }
            QFrame:hover {
                border-color: #1a73e8;
                background-color: #f8f9fa;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Header with name and type
        header_layout = QHBoxLayout()
        
        # Dataset name
        name = self.dataset_info.get('name', 'Unnamed Dataset')
        self.name_label = QLabel(f"<b>{name}</b>")
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("color: #1a73e8; font-size: 11pt;")
        header_layout.addWidget(self.name_label, 1)
        
        # Dataset type badge
        dataset_type = self.dataset_info.get('type', 'Unknown')
        type_badge = QLabel(dataset_type)
        type_badge.setStyleSheet("""
            QLabel {
                background-color: #e8f0fe;
                color: #1a73e8;
                padding: 2px 6px;
                border-radius: 8px;
                font-size: 8pt;
                font-weight: bold;
            }
        """)
        type_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(type_badge)
        
        layout.addLayout(header_layout)
        
        # Compact metadata row
        metadata_layout = QHBoxLayout()
        
        # Owner
        owner = self.dataset_info.get('owner', 'Unknown')
        owner_label = QLabel(f"👤 {owner}")
        owner_label.setStyleSheet("color: #666; font-size: 8pt;")
        metadata_layout.addWidget(owner_label)
        
        metadata_layout.addStretch()
        
        # Row count (if available) - more compact
        row_count = self.dataset_info.get('row_count')
        if row_count is not None:
            if row_count >= 1000000:
                count_str = f"{row_count // 1000000}M"
            elif row_count >= 1000:
                count_str = f"{row_count // 1000}K"
            else:
                count_str = str(row_count)
            count_label = QLabel(f"📊 {count_str}")
            count_label.setStyleSheet("color: #666; font-size: 8pt;")
            metadata_layout.addWidget(count_label)
            
        layout.addLayout(metadata_layout)
        
        # Access and action buttons
        actions_layout = QHBoxLayout()
        
        # Access level indicator
        access_level = self.dataset_info.get('access_level', 'Unknown')
        has_access = access_level in ['Full', 'Read', 'Read-Only']
        
        if has_access:
            access_color = '#4caf50' if access_level == 'Full' else '#ff9800'
            access_text = f"✓ {access_level}"
        else:
            access_color = '#f44336'
            access_text = "✗ No Access"
            
        access_label = QLabel(access_text)
        access_label.setStyleSheet(f"""
            QLabel {{
                color: {access_color};
                font-size: 8pt;
                font-weight: bold;
                padding: 2px 4px;
            }}
        """)
        actions_layout.addWidget(access_label)
        
        actions_layout.addStretch()
        
        # Action buttons
        if has_access and self.dataset_info.get('supports_chat', True):
            chat_btn = QPushButton("Chat")
            chat_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1a73e8;
                    color: white;
                    border: none;
                    padding: 3px 8px;
                    border-radius: 3px;
                    font-size: 8pt;
                }
                QPushButton:hover { background-color: #1557b0; }
            """)
            chat_btn.clicked.connect(self.on_chat_clicked)
            actions_layout.addWidget(chat_btn)
        
        view_btn = QPushButton("View" if has_access else "Request")
        btn_color = "#34a853" if has_access else "#ff9800"
        btn_hover = "#2d8f47" if has_access else "#e68900"
        
        view_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {btn_color};
                color: white;
                border: none;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 8pt;
            }}
            QPushButton:hover {{ background-color: {btn_hover}; }}
        """)
        
        if has_access:
            view_btn.clicked.connect(self.on_view_clicked)
        else:
            view_btn.clicked.connect(self.on_request_access_clicked)
            
        actions_layout.addWidget(view_btn)
        
        layout.addLayout(actions_layout)
        
        # Make the whole item clickable
        self.mousePressEvent = self.on_item_clicked
        
    def on_item_clicked(self, event):
        """Handle item click"""
        self.dataset_selected.emit(self.dataset_info)
        
    def on_chat_clicked(self):
        """Handle chat button click"""
        self.chat_requested.emit(self.dataset_info)
        
    def on_view_clicked(self):
        """Handle view button click"""
        self.dataset_selected.emit(self.dataset_info)
        
    def on_request_access_clicked(self):
        """Handle request access button click"""
        self.access_requested.emit(self.dataset_info)

class DatasetBrowser(QWidget):
    """Dataset browser widget"""
    
    dataset_selected = pyqtSignal(dict)  # dataset_info
    chat_requested = pyqtSignal(dict)  # dataset_info
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.all_datasets = []
        self.filtered_datasets = []
        self.current_filters = {}
        
        self.setup_ui()
        self.setup_connections()
        self.load_datasets()
        
    def setup_ui(self):
        """Setup the dataset browser UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Datasets")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("Refresh datasets")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                background: white;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Search and filters
        search_layout = QVBoxLayout()
        
        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search datasets...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 9pt;
            }
        """)
        search_layout.addWidget(self.search_input)
        
        # Filter row
        filter_layout = QHBoxLayout()
        
        # Type filter
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All Types", "Table", "View", "File", "API", "Stream"])
        self.type_filter.setStyleSheet("font-size: 8pt;")
        filter_layout.addWidget(self.type_filter)
        
        # Access filter
        self.access_filter = QComboBox()
        self.access_filter.addItems(["All Access", "Full Access", "Read Access", "No Access"])
        self.access_filter.setStyleSheet("font-size: 8pt;")
        filter_layout.addWidget(self.access_filter)
        
        search_layout.addLayout(filter_layout)
        layout.addLayout(search_layout)
        
        # Dataset list
        self.dataset_scroll = QScrollArea()
        self.dataset_scroll.setWidgetResizable(True)
        self.dataset_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.dataset_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dataset_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
        """)
        
        self.datasets_container = QWidget()
        self.datasets_layout = QVBoxLayout(self.datasets_container)
        self.datasets_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.datasets_layout.setSpacing(2)
        
        self.dataset_scroll.setWidget(self.datasets_container)
        layout.addWidget(self.dataset_scroll)
        
        # Status label
        self.status_label = QLabel("Loading datasets...")
        self.status_label.setStyleSheet("font-size: 8pt; color: #666; padding: 4px;")
        layout.addWidget(self.status_label)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.search_input.textChanged.connect(self.apply_filters)
        self.type_filter.currentTextChanged.connect(self.apply_filters)
        self.access_filter.currentTextChanged.connect(self.apply_filters)
        
    def load_datasets(self):
        """Load datasets from configuration and external sources"""
        # Load registered datasets
        registered = self.config_manager.get_registered_datasets()
        
        # Add some mock datasets for demonstration
        mock_datasets = [
            {
                'name': 'Customer Analytics',
                'type': 'Table',
                'description': 'Comprehensive customer data including demographics, purchase history, and behavior analytics.',
                'owner': 'Analytics Team',
                'access_level': 'Full',
                'last_updated': '2024-12-20',
                'row_count': 125000,
                'source': 'bigquery://analytics.customers',
                'supports_chat': True,
                'tags': ['customer', 'analytics', 'pii']
            },
            {
                'name': 'Sales Performance',
                'type': 'View',
                'description': 'Aggregated sales metrics by region, product, and time period.',
                'owner': 'Sales Team',
                'access_level': 'Read-Only',
                'last_updated': '2024-12-21',
                'row_count': 50000,
                'source': 'bigquery://sales.performance_view',
                'supports_chat': True,
                'tags': ['sales', 'metrics']
            },
            {
                'name': 'Financial Reports',
                'type': 'File',
                'description': 'Monthly and quarterly financial reports in Excel format.',
                'owner': 'Finance Team',
                'access_level': 'No Access',
                'last_updated': '2024-12-15',
                'source': '/finance/reports/',
                'supports_chat': False,
                'tags': ['finance', 'confidential']
            },
            {
                'name': 'Product Catalog',
                'type': 'API',
                'description': 'Real-time product information including pricing, inventory, and specifications.',
                'owner': 'Product Team',
                'access_level': 'Read',
                'last_updated': '2024-12-21',
                'row_count': 15000,
                'source': 'api://catalog.product.com/v2',
                'supports_chat': True,
                'tags': ['product', 'catalog', 'realtime']
            },
            {
                'name': 'User Activity Stream',
                'type': 'Stream',
                'description': 'Real-time user activity events from web and mobile applications.',
                'owner': 'Engineering Team',
                'access_level': 'Full',
                'last_updated': '2024-12-21',
                'source': 'kafka://events.user_activity',
                'supports_chat': True,
                'tags': ['events', 'realtime', 'user']
            }
        ]
        
        self.all_datasets = registered + mock_datasets
        self.apply_filters()
    
    def apply_dataset_filters(self, filters: Dict[str, Any]):
        """Apply filters to the dataset list"""
        self.current_filters = filters
        self.apply_filters()
        
    def apply_filters(self):
        """Apply current filters to dataset list"""
        search_text = self.search_input.text().lower()
        type_filter = self.type_filter.currentText()
        access_filter = self.access_filter.currentText()
        
        # Get additional filters from filter dialog
        name_filter = self.current_filters.get('name_contains', '').lower()
        owner_filter = self.current_filters.get('owner_contains', '').lower()
        allowed_types = self.current_filters.get('types', [])
        allowed_access = self.current_filters.get('access_levels', [])
        include_tags = self.current_filters.get('include_tags', [])
        exclude_tags = self.current_filters.get('exclude_tags', [])
        min_quality = self.current_filters.get('min_quality_score', 0)
        supports_chat = self.current_filters.get('supports_chat', None)
        has_embeddings = self.current_filters.get('has_embeddings', None)
        has_documentation = self.current_filters.get('has_documentation', None)
        
        self.filtered_datasets = []
        
        for dataset in self.all_datasets:
            # Text search (existing)
            if search_text:
                searchable_text = f"{dataset.get('name', '')} {dataset.get('description', '')} {dataset.get('owner', '')}".lower()
                if search_text not in searchable_text:
                    continue
                    
            # Type filter (existing)
            if type_filter != "All Types":
                if dataset.get('type', '') != type_filter:
                    continue
                    
            # Access filter (existing)
            if access_filter != "All Access":
                access_level = dataset.get('access_level', '')
                if access_filter == "Full Access" and access_level != "Full":
                    continue
                elif access_filter == "Read Access" and access_level not in ["Read", "Read-Only"]:
                    continue
                elif access_filter == "No Access" and access_level != "No Access":
                    continue
            
            # Advanced filters from filter dialog
            
            # Name filter
            if name_filter and name_filter not in dataset.get('name', '').lower():
                continue
                
            # Owner filter
            if owner_filter and owner_filter not in dataset.get('owner', '').lower():
                continue
                
            # Dataset type filter (from advanced dialog)
            if allowed_types and dataset.get('type', '') not in allowed_types:
                continue
                
            # Access level filter (from advanced dialog)
            if allowed_access and dataset.get('access_level', '') not in allowed_access:
                continue
                
            # Tag filters
            dataset_tags = dataset.get('tags', [])
            if include_tags:
                if not any(tag in dataset_tags for tag in include_tags):
                    continue
                    
            if exclude_tags:
                if any(tag in dataset_tags for tag in exclude_tags):
                    continue
                    
            # Quality filter
            dataset_quality = dataset.get('quality_score', 0)
            if dataset_quality < min_quality:
                continue
                
            # AI capabilities filters
            if supports_chat is not None:
                if dataset.get('supports_chat', False) != supports_chat:
                    continue
                    
            if has_embeddings is not None:
                if dataset.get('has_embeddings', False) != has_embeddings:
                    continue
                    
            if has_documentation is not None:
                if dataset.get('has_documentation', False) != has_documentation:
                    continue
                    
            self.filtered_datasets.append(dataset)
            
        self.display_datasets()

    def display_datasets(self):
        """Display filtered datasets"""
        # Clear existing items
        while self.datasets_layout.count():
            child = self.datasets_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Add filtered datasets
        for dataset in self.filtered_datasets:
            dataset_item = DatasetItem(dataset)
            dataset_item.dataset_selected.connect(self.dataset_selected.emit)
            dataset_item.chat_requested.connect(self.chat_requested.emit)
            dataset_item.access_requested.connect(self.handle_access_request)
            self.datasets_layout.addWidget(dataset_item)
            
        # Add stretch to push items to top
        self.datasets_layout.addStretch()
        
        # Update status
        total_count = len(self.all_datasets)
        filtered_count = len(self.filtered_datasets)
        
        if filtered_count != total_count:
            self.status_label.setText(f"Showing {filtered_count} of {total_count} datasets")
        else:
            self.status_label.setText(f"Found {total_count} datasets")
            
    def handle_access_request(self, dataset_info: Dict[str, Any]):
        """Handle access request for a dataset"""
        dialog = AccessRequestDialog(dataset_info, self)
        if dialog.exec():
            # In a real implementation, this would send the request
            QMessageBox.information(
                self, 
                "Request Sent", 
                f"Access request for '{dataset_info['name']}' has been sent to {dataset_info['owner']}."
            )
            
    def refresh(self):
        """Refresh dataset list"""
        self.load_datasets()
        
    def get_selected_datasets(self) -> List[Dict[str, Any]]:
        """Get currently filtered/selected datasets"""
        return self.filtered_datasets.copy()

class AccessRequestDialog(QDialog):
    """Dialog for requesting access to a dataset"""
    
    def __init__(self, dataset_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.dataset_info = dataset_info
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the access request dialog"""
        self.setWindowTitle("Request Dataset Access")
        self.setModal(True)
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Dataset info
        info_label = QLabel(f"<h3>Request access to: {self.dataset_info['name']}</h3>")
        layout.addWidget(info_label)
        
        owner_label = QLabel(f"<b>Owner:</b> {self.dataset_info['owner']}")
        layout.addWidget(owner_label)
        
        desc_label = QLabel(f"<b>Description:</b> {self.dataset_info['description']}")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Access level requested
        form_layout = QFormLayout()
        
        self.access_level = QComboBox()
        self.access_level.addItems(["Read-Only", "Read-Write", "Full Access"])
        form_layout.addRow("Access Level:", self.access_level)
        
        # Business justification
        self.justification = QTextEdit()
        self.justification.setPlaceholderText("Please explain why you need access to this dataset...")
        self.justification.setMaximumHeight(100)
        form_layout.addRow("Justification:", self.justification)
        
        # Duration
        self.duration = QComboBox()
        self.duration.addItems(["30 days", "90 days", "6 months", "1 year", "Permanent"])
        form_layout.addRow("Duration:", self.duration)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        
        request_btn = QPushButton("Send Request")
        request_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        request_btn.clicked.connect(self.accept)
        button_layout.addWidget(request_btn)
        
        layout.addWidget(QFrame())  # Spacer
        layout.addLayout(button_layout)
        
    def get_request_data(self) -> Dict[str, Any]:
        """Get the access request data"""
        return {
            'dataset': self.dataset_info,
            'access_level': self.access_level.currentText(),
            'justification': self.justification.toPlainText(),
            'duration': self.duration.currentText()
        }
