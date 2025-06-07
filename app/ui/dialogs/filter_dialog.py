"""
Filter Dialog
Advanced filtering options for datasets and search results
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QCheckBox,
    QComboBox, QLineEdit, QPushButton, QGroupBox, QLabel,
    QListWidget, QListWidgetItem, QDateEdit, QSpinBox, QSlider
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal
from PyQt6.QtGui import QFont
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FilterDialog(QDialog):
    """Advanced filter dialog for datasets and search results"""
    
    filters_applied = pyqtSignal(dict)  # filter_config
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_config = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the filter dialog UI"""
        self.setWindowTitle("Advanced Filters")
        self.setModal(True)
        self.setMinimumSize(500, 600)
        
        layout = QVBoxLayout(self)
        
        # Dataset Type Filters
        type_group = QGroupBox("Dataset Types")
        type_layout = QVBoxLayout(type_group)
        
        self.type_filters = {}
        dataset_types = ["Table", "View", "File", "API", "Stream", "Model", "Report"]
        
        for dtype in dataset_types:
            checkbox = QCheckBox(dtype)
            checkbox.setChecked(True)  # Default to all selected
            self.type_filters[dtype] = checkbox
            type_layout.addWidget(checkbox)
            
        layout.addWidget(type_group)
        
        # Access Level Filters
        access_group = QGroupBox("Access Levels")
        access_layout = QVBoxLayout(access_group)
        
        self.access_filters = {}
        access_levels = ["Full", "Read-Only", "Read", "No Access", "Pending"]
        
        for access in access_levels:
            checkbox = QCheckBox(access)
            checkbox.setChecked(True)
            self.access_filters[access] = checkbox
            access_layout.addWidget(checkbox)
            
        layout.addWidget(access_group)
        
        # Owner/Team Filters
        owner_group = QGroupBox("Owners & Teams")
        owner_layout = QFormLayout(owner_group)
        
        self.owner_filter = QLineEdit()
        self.owner_filter.setPlaceholderText("Filter by owner name...")
        owner_layout.addRow("Owner Contains:", self.owner_filter)
        
        self.team_filter = QComboBox()
        self.team_filter.addItems([
            "All Teams", "Analytics Team", "Sales Team", "Marketing Team", 
            "Finance Team", "Engineering Team", "Product Team", "Operations Team"
        ])
        owner_layout.addRow("Team:", self.team_filter)
        
        layout.addWidget(owner_group)
        
        # Date Filters
        date_group = QGroupBox("Date Filters")
        date_layout = QFormLayout(date_group)
        
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        date_layout.addRow("Modified After:", self.date_from)
        
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        date_layout.addRow("Modified Before:", self.date_to)
        
        layout.addWidget(date_group)
        
        # Size/Scale Filters
        size_group = QGroupBox("Dataset Size")
        size_layout = QFormLayout(size_group)
        
        self.min_rows = QSpinBox()
        self.min_rows.setRange(0, 999999999)
        self.min_rows.setValue(0)
        self.min_rows.setSuffix(" rows")
        size_layout.addRow("Minimum Rows:", self.min_rows)
        
        self.max_rows = QSpinBox()
        self.max_rows.setRange(0, 999999999)
        self.max_rows.setValue(999999999)
        self.max_rows.setSuffix(" rows")
        size_layout.addRow("Maximum Rows:", self.max_rows)
        
        layout.addWidget(size_group)
        
        # Quality/Rating Filters
        quality_group = QGroupBox("Data Quality")
        quality_layout = QFormLayout(quality_group)
        
        self.quality_score = QSlider(Qt.Orientation.Horizontal)
        self.quality_score.setRange(0, 100)
        self.quality_score.setValue(0)
        self.quality_score.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_score.setTickInterval(20)
        
        quality_label = QLabel("0%")
        self.quality_score.valueChanged.connect(
            lambda v: quality_label.setText(f"{v}%")
        )
        
        quality_row = QHBoxLayout()
        quality_row.addWidget(self.quality_score)
        quality_row.addWidget(quality_label)
        
        quality_layout.addRow("Min Quality Score:", quality_row)
        
        # Data freshness
        self.freshness_filter = QComboBox()
        self.freshness_filter.addItems([
            "Any", "Last 24 hours", "Last 7 days", "Last 30 days", 
            "Last 90 days", "Last year"
        ])
        quality_layout.addRow("Data Freshness:", self.freshness_filter)
        
        layout.addWidget(quality_group)
        
        # Tags Filter
        tags_group = QGroupBox("Tags")
        tags_layout = QVBoxLayout(tags_group)
        
        # Include tags
        include_layout = QHBoxLayout()
        include_layout.addWidget(QLabel("Include Tags:"))
        
        self.include_tags = QLineEdit()
        self.include_tags.setPlaceholderText("customer, analytics, pii (comma-separated)")
        include_layout.addWidget(self.include_tags)
        
        tags_layout.addLayout(include_layout)
        
        # Exclude tags
        exclude_layout = QHBoxLayout()
        exclude_layout.addWidget(QLabel("Exclude Tags:"))
        
        self.exclude_tags = QLineEdit()
        self.exclude_tags.setPlaceholderText("deprecated, test, temp (comma-separated)")
        exclude_layout.addWidget(self.exclude_tags)
        
        tags_layout.addLayout(exclude_layout)
        
        layout.addWidget(tags_group)
        
        # AI/Chat Capabilities
        ai_group = QGroupBox("AI Capabilities")
        ai_layout = QVBoxLayout(ai_group)
        
        self.supports_chat = QCheckBox("Supports AI Chat")
        self.supports_chat.setChecked(True)
        ai_layout.addWidget(self.supports_chat)
        
        self.has_embeddings = QCheckBox("Has Vector Embeddings")
        ai_layout.addWidget(self.has_embeddings)
        
        self.has_documentation = QCheckBox("Has Documentation")
        ai_layout.addWidget(self.has_documentation)
        
        layout.addWidget(ai_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_filters)
        button_layout.addWidget(clear_btn)
        
        preset_btn = QPushButton("Presets")
        preset_btn.clicked.connect(self.show_presets)
        button_layout.addWidget(preset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply Filters")
        apply_btn.setStyleSheet("""
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
        apply_btn.clicked.connect(self.apply_filters)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
        
    def get_filters(self) -> Dict[str, Any]:
        """Get current filter configuration"""
        filters = {
            'types': [dtype for dtype, checkbox in self.type_filters.items() if checkbox.isChecked()],
            'access_levels': [access for access, checkbox in self.access_filters.items() if checkbox.isChecked()],
            'owner_contains': self.owner_filter.text(),
            'team': self.team_filter.currentText() if self.team_filter.currentText() != "All Teams" else None,
            'date_from': self.date_from.date().toPython(),
            'date_to': self.date_to.date().toPython(),
            'min_rows': self.min_rows.value(),
            'max_rows': self.max_rows.value() if self.max_rows.value() < 999999999 else None,
            'min_quality_score': self.quality_score.value(),
            'freshness': self.freshness_filter.currentText() if self.freshness_filter.currentText() != "Any" else None,
            'include_tags': [tag.strip() for tag in self.include_tags.text().split(',') if tag.strip()],
            'exclude_tags': [tag.strip() for tag in self.exclude_tags.text().split(',') if tag.strip()],
            'supports_chat': self.supports_chat.isChecked(),
            'has_embeddings': self.has_embeddings.isChecked(),
            'has_documentation': self.has_documentation.isChecked()
        }
        
        return filters
        
    def set_filters(self, filters: Dict[str, Any]):
        """Set filter configuration"""
        # Dataset types
        for dtype, checkbox in self.type_filters.items():
            checkbox.setChecked(dtype in filters.get('types', []))
            
        # Access levels
        for access, checkbox in self.access_filters.items():
            checkbox.setChecked(access in filters.get('access_levels', []))
            
        # Owner and team
        self.owner_filter.setText(filters.get('owner_contains', ''))
        if filters.get('team'):
            self.team_filter.setCurrentText(filters['team'])
            
        # Dates
        if filters.get('date_from'):
            self.date_from.setDate(QDate.fromString(str(filters['date_from']), Qt.DateFormat.ISODate))
        if filters.get('date_to'):
            self.date_to.setDate(QDate.fromString(str(filters['date_to']), Qt.DateFormat.ISODate))
            
        # Size
        self.min_rows.setValue(filters.get('min_rows', 0))
        if filters.get('max_rows'):
            self.max_rows.setValue(filters['max_rows'])
            
        # Quality
        self.quality_score.setValue(filters.get('min_quality_score', 0))
        if filters.get('freshness'):
            self.freshness_filter.setCurrentText(filters['freshness'])
            
        # Tags
        if filters.get('include_tags'):
            self.include_tags.setText(', '.join(filters['include_tags']))
        if filters.get('exclude_tags'):
            self.exclude_tags.setText(', '.join(filters['exclude_tags']))
            
        # AI capabilities
        self.supports_chat.setChecked(filters.get('supports_chat', True))
        self.has_embeddings.setChecked(filters.get('has_embeddings', False))
        self.has_documentation.setChecked(filters.get('has_documentation', False))
        
    def clear_filters(self):
        """Clear all filters to default state"""
        # Check all type and access filters
        for checkbox in self.type_filters.values():
            checkbox.setChecked(True)
        for checkbox in self.access_filters.values():
            checkbox.setChecked(True)
            
        # Clear text fields
        self.owner_filter.clear()
        self.team_filter.setCurrentIndex(0)
        
        # Reset dates
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_to.setDate(QDate.currentDate())
        
        # Reset size
        self.min_rows.setValue(0)
        self.max_rows.setValue(999999999)
        
        # Reset quality
        self.quality_score.setValue(0)
        self.freshness_filter.setCurrentIndex(0)
        
        # Clear tags
        self.include_tags.clear()
        self.exclude_tags.clear()
        
        # Reset AI capabilities
        self.supports_chat.setChecked(True)
        self.has_embeddings.setChecked(False)
        self.has_documentation.setChecked(False)
        
    def show_presets(self):
        """Show filter presets dialog"""
        presets_dialog = FilterPresetsDialog(self)
        if presets_dialog.exec():
            preset = presets_dialog.get_selected_preset()
            if preset:
                self.set_filters(preset)
                
    def apply_filters(self):
        """Apply current filters"""
        filters = self.get_filters()
        self.filter_config = filters
        self.filters_applied.emit(filters)
        self.accept()

class FilterPresetsDialog(QDialog):
    """Dialog for managing filter presets"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_preset = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup presets dialog UI"""
        self.setWindowTitle("Filter Presets")
        self.setModal(True)
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Presets list
        self.presets_list = QListWidget()
        self.populate_presets()
        layout.addWidget(self.presets_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        select_btn = QPushButton("Select Preset")
        select_btn.clicked.connect(self.select_preset)
        button_layout.addWidget(select_btn)
        
        layout.addLayout(button_layout)
        
    def populate_presets(self):
        """Populate the presets list"""
        presets = self.get_default_presets()
        
        for name, description in presets.items():
            item = QListWidgetItem(f"{name}\n{description}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.presets_list.addItem(item)
            
    def get_default_presets(self) -> Dict[str, str]:
        """Get default filter presets"""
        return {
            "Recent & Accessible": "Show only datasets modified in last 30 days with full or read access",
            "High Quality Data": "Show only high-quality datasets (>80% quality score) with documentation",
            "AI-Ready Datasets": "Show datasets that support AI chat and have vector embeddings",
            "My Team's Data": "Show datasets owned by my team with full access",
            "Large Datasets": "Show datasets with more than 100K rows",
            "Analytics Focus": "Show analytics-related datasets (tables and views only)",
            "External APIs": "Show external API data sources",
            "Needs Attention": "Show datasets with low quality scores or no documentation"
        }
        
    def get_preset_filters(self, preset_name: str) -> Dict[str, Any]:
        """Get filter configuration for a preset"""
        presets = {
            "Recent & Accessible": {
                'types': ['Table', 'View', 'File', 'API', 'Stream'],
                'access_levels': ['Full', 'Read-Only', 'Read'],
                'freshness': 'Last 30 days',
                'supports_chat': True
            },
            "High Quality Data": {
                'min_quality_score': 80,
                'has_documentation': True,
                'types': ['Table', 'View'],
                'access_levels': ['Full', 'Read-Only', 'Read']
            },
            "AI-Ready Datasets": {
                'supports_chat': True,
                'has_embeddings': True,
                'access_levels': ['Full', 'Read-Only', 'Read']
            },
            "Large Datasets": {
                'min_rows': 100000,
                'types': ['Table', 'View']
            },
            "Analytics Focus": {
                'types': ['Table', 'View'],
                'include_tags': ['analytics', 'reporting', 'metrics']
            },
            "External APIs": {
                'types': ['API'],
                'access_levels': ['Full', 'Read-Only', 'Read']
            }
        }
        
        return presets.get(preset_name, {})
        
    def select_preset(self):
        """Select the chosen preset"""
        current_item = self.presets_list.currentItem()
        if current_item:
            preset_name = current_item.data(Qt.ItemDataRole.UserRole)
            self.selected_preset = self.get_preset_filters(preset_name)
            self.accept()
            
    def get_selected_preset(self) -> Dict[str, Any]:
        """Get the selected preset configuration"""
        return self.selected_preset
