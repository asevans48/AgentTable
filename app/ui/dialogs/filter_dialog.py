"""
Filter Dialog
Advanced filtering options for datasets and search results with improved organization
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QCheckBox,
    QComboBox, QLineEdit, QPushButton, QGroupBox, QLabel,
    QListWidget, QListWidgetItem, QDateEdit, QSpinBox, QSlider,
    QTabWidget, QWidget, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal
from PyQt6.QtGui import QFont
from typing import Dict, List, Any
from datetime import date
import logging

logger = logging.getLogger(__name__)

class BasicFiltersTab(QWidget):
    """Basic, commonly used filters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup basic filters UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Quick Filter Presets
        presets_group = QGroupBox("Quick Filters")
        presets_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        presets_layout = QVBoxLayout(presets_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "All Datasets",
            "My Accessible Data",
            "Recent & High Quality",
            "AI-Ready Datasets",
            "Large Datasets (100K+ rows)",
            "Team Datasets",
            "Custom Filter..."
        ])
        presets_layout.addWidget(self.preset_combo)
        
        layout.addWidget(presets_group)
        
        # Dataset Types - More compact
        types_group = QGroupBox("Dataset Types")
        types_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        types_layout = QVBoxLayout(types_group)
        
        # Organize types in rows
        types_row1 = QHBoxLayout()
        types_row2 = QHBoxLayout()
        
        self.type_filters = {}
        dataset_types = ["Table", "View", "File", "API", "Stream", "Model", "Report"]
        
        for i, dtype in enumerate(dataset_types):
            checkbox = QCheckBox(dtype)
            checkbox.setChecked(True)
            self.type_filters[dtype] = checkbox
            
            if i < 4:
                types_row1.addWidget(checkbox)
            else:
                types_row2.addWidget(checkbox)
                
        types_row1.addStretch()
        types_row2.addStretch()
        
        types_layout.addLayout(types_row1)
        types_layout.addLayout(types_row2)
        
        layout.addWidget(types_group)
        
        # Access Levels - More compact
        access_group = QGroupBox("Access Permissions")
        access_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        access_layout = QHBoxLayout(access_group)
        
        self.access_filters = {}
        access_levels = ["Full", "Read-Only", "Read", "No Access", "Pending"]
        
        for access in access_levels:
            checkbox = QCheckBox(access)
            checkbox.setChecked(True)
            self.access_filters[access] = checkbox
            access_layout.addWidget(checkbox)
            
        access_layout.addStretch()
        layout.addWidget(access_group)
        
        # Search by Name/Owner
        search_group = QGroupBox("Text Search")
        search_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        search_layout = QFormLayout(search_group)
        
        self.name_filter = QLineEdit()
        self.name_filter.setPlaceholderText("Filter by dataset name...")
        search_layout.addRow("Dataset Name:", self.name_filter)
        
        self.owner_filter = QLineEdit()
        self.owner_filter.setPlaceholderText("Filter by owner name...")
        search_layout.addRow("Owner:", self.owner_filter)
        
        layout.addWidget(search_group)
        
        layout.addStretch()

class AdvancedFiltersTab(QWidget):
    """Advanced filtering options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup advanced filters UI"""
        # Use scroll area for advanced options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)
        
        # Date Filters
        date_group = QGroupBox("Date Range")
        date_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
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
        
        # Size Filters
        size_group = QGroupBox("Dataset Size")
        size_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        size_layout = QFormLayout(size_group)
        
        # Size range with better formatting
        size_row = QHBoxLayout()
        
        self.min_rows = QComboBox()
        self.min_rows.addItems(["Any", "1K+", "10K+", "100K+", "1M+", "10M+"])
        size_row.addWidget(QLabel("Min:"))
        size_row.addWidget(self.min_rows)
        
        size_row.addWidget(QLabel(" to "))
        
        self.max_rows = QComboBox()
        self.max_rows.addItems(["Any", "1K", "10K", "100K", "1M", "10M", "100M+"])
        size_row.addWidget(QLabel("Max:"))
        size_row.addWidget(self.max_rows)
        
        size_row.addStretch()
        size_layout.addRow("Row Count:", size_row)
        
        layout.addWidget(size_group)
        
        # Quality Filters
        quality_group = QGroupBox("Data Quality")
        quality_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        quality_layout = QFormLayout(quality_group)
        
        # Quality score with better layout
        quality_row = QHBoxLayout()
        
        self.quality_score = QSlider(Qt.Orientation.Horizontal)
        self.quality_score.setRange(0, 100)
        self.quality_score.setValue(0)
        self.quality_score.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_score.setTickInterval(25)
        
        self.quality_label = QLabel("0%")
        self.quality_score.valueChanged.connect(
            lambda v: self.quality_label.setText(f"{v}%")
        )
        
        quality_row.addWidget(self.quality_score)
        quality_row.addWidget(self.quality_label)
        
        quality_layout.addRow("Min Quality Score:", quality_row)
        
        # Data freshness
        self.freshness_filter = QComboBox()
        self.freshness_filter.addItems([
            "Any", "Last 24 hours", "Last 7 days", "Last 30 days", 
            "Last 90 days", "Last year"
        ])
        quality_layout.addRow("Data Freshness:", self.freshness_filter)
        
        layout.addWidget(quality_group)
        
        # Team and Organization
        team_group = QGroupBox("Team & Organization")
        team_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        team_layout = QFormLayout(team_group)
        
        self.team_filter = QComboBox()
        self.team_filter.addItems([
            "All Teams", "Analytics Team", "Sales Team", "Marketing Team", 
            "Finance Team", "Engineering Team", "Product Team", "Operations Team"
        ])
        team_layout.addRow("Team:", self.team_filter)
        
        layout.addWidget(team_group)
        
        scroll.setWidget(scroll_widget)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)

class TagsAndAITab(QWidget):
    """Tags and AI capabilities filters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup tags and AI filters UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Tags Filter
        tags_group = QGroupBox("Tags")
        tags_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        tags_layout = QVBoxLayout(tags_group)
        
        # Include tags
        include_layout = QFormLayout()
        self.include_tags = QLineEdit()
        self.include_tags.setPlaceholderText("customer, analytics, pii (comma-separated)")
        include_layout.addRow("Include Tags:", self.include_tags)
        
        # Exclude tags  
        self.exclude_tags = QLineEdit()
        self.exclude_tags.setPlaceholderText("deprecated, test, temp (comma-separated)")
        include_layout.addRow("Exclude Tags:", self.exclude_tags)
        
        tags_layout.addLayout(include_layout)
        
        # Popular tags as checkboxes
        popular_label = QLabel("Popular Tags:")
        popular_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        tags_layout.addWidget(popular_label)
        
        popular_tags_layout = QHBoxLayout()
        popular_tags = ["customer", "analytics", "finance", "sales", "marketing", "pii"]
        
        self.popular_tag_checkboxes = {}
        for tag in popular_tags:
            checkbox = QCheckBox(tag)
            self.popular_tag_checkboxes[tag] = checkbox
            popular_tags_layout.addWidget(checkbox)
            
        popular_tags_layout.addStretch()
        tags_layout.addLayout(popular_tags_layout)
        
        layout.addWidget(tags_group)
        
        # AI/Chat Capabilities
        ai_group = QGroupBox("AI & Analytics Capabilities")
        ai_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        ai_layout = QVBoxLayout(ai_group)
        
        ai_row1 = QHBoxLayout()
        ai_row2 = QHBoxLayout()
        
        self.supports_chat = QCheckBox("AI Chat Support")
        self.supports_chat.setChecked(True)
        ai_row1.addWidget(self.supports_chat)
        
        self.has_embeddings = QCheckBox("Vector Embeddings")
        ai_row1.addWidget(self.has_embeddings)
        
        self.has_documentation = QCheckBox("Has Documentation")
        ai_row2.addWidget(self.has_documentation)
        
        self.is_queryable = QCheckBox("SQL Queryable")
        ai_row2.addWidget(self.is_queryable)
        
        ai_row1.addStretch()
        ai_row2.addStretch()
        
        ai_layout.addLayout(ai_row1)
        ai_layout.addLayout(ai_row2)
        
        layout.addWidget(ai_group)
        
        layout.addStretch()

class FilterDialog(QDialog):
    """Advanced filter dialog with improved organization"""
    
    filters_applied = pyqtSignal(dict)  # filter_config
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_config = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the filter dialog UI"""
        self.setWindowTitle("Dataset Filters")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Filter Datasets")
        header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_label.setStyleSheet("color: #333; padding: 10px;")
        layout.addWidget(header_label)
        
        # Tab widget for organized filters
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
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
                background-color: #e6e6e6;
            }
        """)
        
        # Add tabs
        self.basic_tab = BasicFiltersTab()
        self.tab_widget.addTab(self.basic_tab, "📊 Basic Filters")
        
        self.advanced_tab = AdvancedFiltersTab()
        self.tab_widget.addTab(self.advanced_tab, "⚙️ Advanced")
        
        self.tags_ai_tab = TagsAndAITab()
        self.tab_widget.addTab(self.tags_ai_tab, "🏷️ Tags & AI")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons with better styling
        button_frame = QFrame()
        button_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        button_frame.setStyleSheet("background-color: #f8f9fa; border-top: 1px solid #dee2e6;")
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(10, 10, 10, 10)
        
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self.clear_filters)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(clear_btn)
        
        preset_btn = QPushButton("📋 Load Preset")
        preset_btn.clicked.connect(self.show_presets)
        preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        button_layout.addWidget(preset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("✅ Apply Filters")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        apply_btn.clicked.connect(self.apply_filters)
        button_layout.addWidget(apply_btn)
        
        layout.addWidget(button_frame)
        
    def get_filters(self) -> Dict[str, Any]:
        """Get current filter configuration from all tabs"""
        filters = {}
        
        # Basic filters
        basic = self.basic_tab
        filters['preset'] = basic.preset_combo.currentText()
        filters['types'] = [dtype for dtype, checkbox in basic.type_filters.items() if checkbox.isChecked()]
        filters['access_levels'] = [access for access, checkbox in basic.access_filters.items() if checkbox.isChecked()]
        filters['name_contains'] = basic.name_filter.text()
        filters['owner_contains'] = basic.owner_filter.text()
        
        # Advanced filters
        advanced = self.advanced_tab
        # Convert QDate to Python date - compatible across PyQt6 versions
        date_from = advanced.date_from.date()
        date_to = advanced.date_to.date()
        
        # Try different conversion methods based on PyQt6 version
        try:
            filters['date_from'] = date_from.toPyDate()
            filters['date_to'] = date_to.toPyDate()
        except AttributeError:
            # Fallback for older PyQt6 versions
            from datetime import date
            filters['date_from'] = date(date_from.year(), date_from.month(), date_from.day())
            filters['date_to'] = date(date_to.year(), date_to.month(), date_to.day())
        
        # Convert size combo selections to numbers
        min_size_map = {"Any": 0, "1K+": 1000, "10K+": 10000, "100K+": 100000, "1M+": 1000000, "10M+": 10000000}
        max_size_map = {"Any": None, "1K": 1000, "10K": 10000, "100K": 100000, "1M": 1000000, "10M": 10000000, "100M+": None}
        
        filters['min_rows'] = min_size_map.get(advanced.min_rows.currentText(), 0)
        filters['max_rows'] = max_size_map.get(advanced.max_rows.currentText(), None)
        filters['min_quality_score'] = advanced.quality_score.value()
        filters['freshness'] = advanced.freshness_filter.currentText() if advanced.freshness_filter.currentText() != "Any" else None
        filters['team'] = advanced.team_filter.currentText() if advanced.team_filter.currentText() != "All Teams" else None
        
        # Tags and AI filters
        tags_ai = self.tags_ai_tab
        filters['include_tags'] = [tag.strip() for tag in tags_ai.include_tags.text().split(',') if tag.strip()]
        filters['exclude_tags'] = [tag.strip() for tag in tags_ai.exclude_tags.text().split(',') if tag.strip()]
        
        # Add popular tags that are checked
        for tag, checkbox in tags_ai.popular_tag_checkboxes.items():
            if checkbox.isChecked() and tag not in filters['include_tags']:
                filters['include_tags'].append(tag)
                
        filters['supports_chat'] = tags_ai.supports_chat.isChecked()
        filters['has_embeddings'] = tags_ai.has_embeddings.isChecked()
        filters['has_documentation'] = tags_ai.has_documentation.isChecked()
        filters['is_queryable'] = tags_ai.is_queryable.isChecked()
        
        return filters
        
    def set_filters(self, filters: Dict[str, Any]):
        """Set filter configuration across all tabs"""
        # Basic filters
        basic = self.basic_tab
        
        if filters.get('preset'):
            basic.preset_combo.setCurrentText(filters['preset'])
            
        # Dataset types
        for dtype, checkbox in basic.type_filters.items():
            checkbox.setChecked(dtype in filters.get('types', []))
            
        # Access levels
        for access, checkbox in basic.access_filters.items():
            checkbox.setChecked(access in filters.get('access_levels', []))
            
        # Text filters
        basic.name_filter.setText(filters.get('name_contains', ''))
        basic.owner_filter.setText(filters.get('owner_contains', ''))
        
        # Advanced filters
        advanced = self.advanced_tab
        
        # Dates
        if filters.get('date_from'):
            date_from = filters['date_from']
            if isinstance(date_from, date):
                advanced.date_from.setDate(QDate(date_from))
            elif isinstance(date_from, str):
                advanced.date_from.setDate(QDate.fromString(date_from, Qt.DateFormat.ISODate))
                
        if filters.get('date_to'):
            date_to = filters['date_to']
            if isinstance(date_to, date):
                advanced.date_to.setDate(QDate(date_to))
            elif isinstance(date_to, str):
                advanced.date_to.setDate(QDate.fromString(date_to, Qt.DateFormat.ISODate))
            
        # Quality
        advanced.quality_score.setValue(filters.get('min_quality_score', 0))
        if filters.get('freshness'):
            advanced.freshness_filter.setCurrentText(filters['freshness'])
        if filters.get('team'):
            advanced.team_filter.setCurrentText(filters['team'])
            
        # Tags and AI
        tags_ai = self.tags_ai_tab
        
        if filters.get('include_tags'):
            tags_ai.include_tags.setText(', '.join(filters['include_tags']))
        if filters.get('exclude_tags'):
            tags_ai.exclude_tags.setText(', '.join(filters['exclude_tags']))
            
        # AI capabilities
        tags_ai.supports_chat.setChecked(filters.get('supports_chat', True))
        tags_ai.has_embeddings.setChecked(filters.get('has_embeddings', False))
        tags_ai.has_documentation.setChecked(filters.get('has_documentation', False))
        tags_ai.is_queryable.setChecked(filters.get('is_queryable', False))
        
    def clear_filters(self):
        """Clear all filters to default state"""
        # Basic tab
        basic = self.basic_tab
        basic.preset_combo.setCurrentIndex(0)
        
        # Check all type and access filters
        for checkbox in basic.type_filters.values():
            checkbox.setChecked(True)
        for checkbox in basic.access_filters.values():
            checkbox.setChecked(True)
            
        # Clear text fields
        basic.name_filter.clear()
        basic.owner_filter.clear()
        
        # Advanced tab
        advanced = self.advanced_tab
        
        # Reset dates
        advanced.date_from.setDate(QDate.currentDate().addDays(-30))
        advanced.date_to.setDate(QDate.currentDate())
        
        # Reset size
        advanced.min_rows.setCurrentIndex(0)  # "Any"
        advanced.max_rows.setCurrentIndex(0)  # "Any"
        
        # Reset quality
        advanced.quality_score.setValue(0)
        advanced.freshness_filter.setCurrentIndex(0)
        advanced.team_filter.setCurrentIndex(0)
        
        # Tags and AI tab
        tags_ai = self.tags_ai_tab
        
        # Clear tags
        tags_ai.include_tags.clear()
        tags_ai.exclude_tags.clear()
        
        # Uncheck popular tags
        for checkbox in tags_ai.popular_tag_checkboxes.values():
            checkbox.setChecked(False)
        
        # Reset AI capabilities
        tags_ai.supports_chat.setChecked(True)
        tags_ai.has_embeddings.setChecked(False)
        tags_ai.has_documentation.setChecked(False)
        tags_ai.is_queryable.setChecked(False)
        
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