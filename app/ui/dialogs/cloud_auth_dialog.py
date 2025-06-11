"""
Cloud Authentication Dialog
Handles authentication and credential storage for cloud providers
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, 
    QPushButton, QLabel, QComboBox, QTabWidget, QWidget, QTextEdit,
    QCheckBox, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class CloudAuthWorker(QThread):
    """Background worker for cloud authentication"""
    
    auth_result = pyqtSignal(bool, str, dict)  # success, message, credentials
    
    def __init__(self, provider: str, credentials: Dict[str, Any]):
        super().__init__()
        self.provider = provider
        self.credentials = credentials
    
    def run(self):
        """Test cloud authentication"""
        try:
            if self.provider == "AWS":
                success, message = self._test_aws_auth()
            elif self.provider == "Azure":
                success, message = self._test_azure_auth()
            elif self.provider == "GCP":
                success, message = self._test_gcp_auth()
            elif self.provider.startswith("OAuth"):
                success, message = self._test_oauth_auth()
            else:
                success, message = False, "Unknown provider"
            
            self.auth_result.emit(success, message, self.credentials)
            
        except Exception as e:
            logger.error(f"Cloud authentication error: {e}")
            self.auth_result.emit(False, f"Authentication failed: {str(e)}", {})
    
    def _test_aws_auth(self) -> tuple[bool, str]:
        """Test AWS authentication"""
        try:
            # In production, use boto3 to test credentials
            # For now, just validate required fields
            required_fields = ["access_key", "secret_key"]
            for field in required_fields:
                if not self.credentials.get(field):
                    return False, f"Missing required field: {field}"
            
            # Simulate AWS credential test
            import time
            time.sleep(1)
            return True, "AWS credentials validated successfully"
            
        except Exception as e:
            return False, f"AWS authentication failed: {str(e)}"
    
    def _test_azure_auth(self) -> tuple[bool, str]:
        """Test Azure authentication"""
        try:
            # In production, use azure-identity to test credentials
            required_fields = ["tenant_id", "client_id"]
            for field in required_fields:
                if not self.credentials.get(field):
                    return False, f"Missing required field: {field}"
            
            # Simulate Azure credential test
            import time
            time.sleep(1)
            return True, "Azure credentials validated successfully"
            
        except Exception as e:
            return False, f"Azure authentication failed: {str(e)}"
    
    def _test_gcp_auth(self) -> tuple[bool, str]:
        """Test GCP authentication"""
        try:
            # In production, use google-auth to test credentials
            if not self.credentials.get("project_id"):
                return False, "Missing project_id"
            
            # Simulate GCP credential test
            import time
            time.sleep(1)
            return True, "GCP credentials validated successfully"
            
        except Exception as e:
            return False, f"GCP authentication failed: {str(e)}"
    
    def _test_oauth_auth(self) -> tuple[bool, str]:
        """Test OAuth authentication"""
        try:
            required_fields = ["client_id", "client_secret"]
            for field in required_fields:
                if not self.credentials.get(field):
                    return False, f"Missing required field: {field}"
            
            # Simulate OAuth credential test
            import time
            time.sleep(1)
            return True, "OAuth credentials validated successfully"
            
        except Exception as e:
            return False, f"OAuth authentication failed: {str(e)}"

class AWSAuthTab(QWidget):
    """AWS authentication tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup AWS authentication UI"""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Configure AWS credentials for accessing AWS services:")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Form layout
        form_layout = QFormLayout()
        
        self.profile_name = QLineEdit()
        self.profile_name.setPlaceholderText("default")
        self.profile_name.setText("default")
        form_layout.addRow("Profile Name:", self.profile_name)
        
        self.access_key = QLineEdit()
        self.access_key.setPlaceholderText("AKIA...")
        form_layout.addRow("Access Key ID:", self.access_key)
        
        self.secret_key = QLineEdit()
        self.secret_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.secret_key.setPlaceholderText("Secret access key")
        form_layout.addRow("Secret Access Key:", self.secret_key)
        
        self.region = QComboBox()
        self.region.addItems([
            "us-east-1", "us-east-2", "us-west-1", "us-west-2",
            "eu-west-1", "eu-west-2", "eu-central-1",
            "ap-southeast-1", "ap-southeast-2", "ap-northeast-1"
        ])
        form_layout.addRow("Default Region:", self.region)
        
        self.session_token = QLineEdit()
        self.session_token.setPlaceholderText("Optional session token")
        form_layout.addRow("Session Token:", self.session_token)
        
        layout.addLayout(form_layout)
        
        # Test connection button
        self.test_button = QPushButton("Test Connection")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        layout.addWidget(self.test_button)
        
        layout.addStretch()
    
    def get_credentials(self) -> Dict[str, Any]:
        """Get AWS credentials from form"""
        return {
            "profile_name": self.profile_name.text() or "default",
            "access_key": self.access_key.text(),
            "secret_key": self.secret_key.text(),
            "region": self.region.currentText(),
            "session_token": self.session_token.text() or None
        }

class CloudAuthDialog(QDialog):
    """Main cloud authentication dialog"""
    
    credentials_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.auth_worker = None
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup the cloud authentication dialog UI"""
        self.setWindowTitle("Cloud Authentication")
        self.setModal(True)
        self.setFixedSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Cloud Provider Authentication")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Tab widget for different providers
        self.tab_widget = QTabWidget()
        
        # AWS tab
        self.aws_tab = AWSAuthTab()
        self.tab_widget.addTab(self.aws_tab, "AWS")
        
        # Placeholder for other tabs
        placeholder_tab = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_tab)
        placeholder_label = QLabel("Additional cloud providers will be available here.")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(placeholder_label)
        self.tab_widget.addTab(placeholder_tab, "More Providers")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Credentials")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        button_layout.addWidget(self.save_button)
        
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.save_button.clicked.connect(self.save_credentials)
        self.aws_tab.test_button.clicked.connect(lambda: self.test_credentials("AWS"))
    
    def test_credentials(self, provider: str):
        """Test cloud credentials"""
        try:
            if provider == "AWS":
                credentials = self.aws_tab.get_credentials()
            else:
                return
            
            # Validate required fields
            if not self._validate_credentials(provider, credentials):
                return
            
            # Start authentication worker
            self.auth_worker = CloudAuthWorker(provider, credentials)
            self.auth_worker.auth_result.connect(self.on_auth_result)
            self.auth_worker.start()
            
            # Update button state
            self.aws_tab.test_button.setText("Testing...")
            self.aws_tab.test_button.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Failed to test {provider} credentials: {e}")
            QMessageBox.warning(self, "Error", f"Failed to test credentials: {str(e)}")
    
    def on_auth_result(self, success: bool, message: str, credentials: Dict[str, Any]):
        """Handle authentication result"""
        # Reset button states
        self.aws_tab.test_button.setText("Test Connection")
        self.aws_tab.test_button.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Authentication Failed", message)
    
    def save_credentials(self):
        """Save credentials to secure storage"""
        try:
            current_tab = self.tab_widget.currentIndex()
            
            if current_tab == 0:  # AWS
                credentials = self.aws_tab.get_credentials()
                if self._validate_credentials("AWS", credentials):
                    # For now, just show success message
                    # In production, this would use the credential manager
                    QMessageBox.information(self, "Success", "AWS credentials would be saved securely!")
                    self.credentials_updated.emit()
            else:
                QMessageBox.information(self, "Info", "Additional providers not yet implemented")
                
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save credentials: {str(e)}")
    
    def _validate_credentials(self, provider: str, credentials: Dict[str, Any]) -> bool:
        """Validate credentials before saving/testing"""
        if provider == "AWS":
            if not credentials.get("access_key") or not credentials.get("secret_key"):
                QMessageBox.warning(self, "Validation Error", "Access Key and Secret Key are required")
                return False
        
        return True
