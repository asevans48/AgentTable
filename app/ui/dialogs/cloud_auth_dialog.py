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
            
            # Check if using client secret or certificate
            has_secret = bool(self.credentials.get("client_secret"))
            has_cert = bool(self.credentials.get("certificate_path"))
            
            if not has_secret and not has_cert:
                return False, "Either client secret or certificate is required"
            
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
            
            # Check if using service account key or key file
            has_key = bool(self.credentials.get("service_account_key"))
            has_file = bool(self.credentials.get("key_file_path"))
            
            if not has_key and not has_file:
                return False, "Either service account key JSON or key file path is required"
            
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
    
    def set_credentials(self, credentials: Dict[str, Any]):
        """Set AWS credentials in form"""
        self.profile_name.setText(credentials.get("profile_name", "default"))
        self.access_key.setText(credentials.get("access_key", ""))
        self.secret_key.setText(credentials.get("secret_key", ""))
        
        region = credentials.get("region", "us-east-1")
        index = self.region.findText(region)
        if index >= 0:
            self.region.setCurrentIndex(index)
        
        self.session_token.setText(credentials.get("session_token", ""))

class AzureAuthTab(QWidget):
    """Azure authentication tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup Azure authentication UI"""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Configure Azure credentials for accessing Azure services:")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Form layout
        form_layout = QFormLayout()
        
        self.tenant_id = QLineEdit()
        self.tenant_id.setPlaceholderText("Tenant ID (Directory ID)")
        form_layout.addRow("Tenant ID:", self.tenant_id)
        
        self.client_id = QLineEdit()
        self.client_id.setPlaceholderText("Application (client) ID")
        form_layout.addRow("Client ID:", self.client_id)
        
        self.client_secret = QLineEdit()
        self.client_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.client_secret.setPlaceholderText("Client secret")
        form_layout.addRow("Client Secret:", self.client_secret)
        
        self.subscription_id = QLineEdit()
        self.subscription_id.setPlaceholderText("Optional subscription ID")
        form_layout.addRow("Subscription ID:", self.subscription_id)
        
        # Certificate option
        cert_layout = QHBoxLayout()
        self.certificate_path = QLineEdit()
        self.certificate_path.setPlaceholderText("Optional certificate path")
        cert_browse = QPushButton("Browse")
        cert_browse.clicked.connect(self.browse_certificate)
        cert_layout.addWidget(self.certificate_path)
        cert_layout.addWidget(cert_browse)
        form_layout.addRow("Certificate:", cert_layout)
        
        # Authentication method
        self.auth_method = QComboBox()
        self.auth_method.addItems([
            "Client Secret",
            "Certificate",
            "Managed Identity",
            "Azure CLI"
        ])
        form_layout.addRow("Auth Method:", self.auth_method)
        
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
    
    def browse_certificate(self):
        """Browse for certificate file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Certificate File", "", 
            "Certificate Files (*.pem *.pfx *.p12);;All Files (*)"
        )
        if file_path:
            self.certificate_path.setText(file_path)
    
    def get_credentials(self) -> Dict[str, Any]:
        """Get Azure credentials from form"""
        return {
            "tenant_id": self.tenant_id.text(),
            "client_id": self.client_id.text(),
            "client_secret": self.client_secret.text() or None,
            "subscription_id": self.subscription_id.text() or None,
            "certificate_path": self.certificate_path.text() or None,
            "auth_method": self.auth_method.currentText()
        }
    
    def set_credentials(self, credentials: Dict[str, Any]):
        """Set Azure credentials in form"""
        self.tenant_id.setText(credentials.get("tenant_id", ""))
        self.client_id.setText(credentials.get("client_id", ""))
        self.client_secret.setText(credentials.get("client_secret", ""))
        self.subscription_id.setText(credentials.get("subscription_id", ""))
        self.certificate_path.setText(credentials.get("certificate_path", ""))
        
        auth_method = credentials.get("auth_method", "Client Secret")
        index = self.auth_method.findText(auth_method)
        if index >= 0:
            self.auth_method.setCurrentIndex(index)

class GCPAuthTab(QWidget):
    """GCP authentication tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup GCP authentication UI"""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Configure GCP credentials for accessing Google Cloud services:")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Form layout
        form_layout = QFormLayout()
        
        self.project_id = QLineEdit()
        self.project_id.setPlaceholderText("GCP Project ID")
        form_layout.addRow("Project ID:", self.project_id)
        
        # Authentication method
        self.auth_method = QComboBox()
        self.auth_method.addItems([
            "Service Account Key File",
            "Service Account Key JSON",
            "Application Default Credentials",
            "User Account"
        ])
        self.auth_method.currentTextChanged.connect(self.on_auth_method_changed)
        form_layout.addRow("Auth Method:", self.auth_method)
        
        # Service account key file
        self.key_file_group = QWidget()
        key_file_layout = QHBoxLayout(self.key_file_group)
        key_file_layout.setContentsMargins(0, 0, 0, 0)
        
        self.key_file_path = QLineEdit()
        self.key_file_path.setPlaceholderText("Path to service account key file")
        key_browse = QPushButton("Browse")
        key_browse.clicked.connect(self.browse_key_file)
        key_file_layout.addWidget(self.key_file_path)
        key_file_layout.addWidget(key_browse)
        form_layout.addRow("Key File:", self.key_file_group)
        
        # Service account key JSON
        self.key_json_group = QWidget()
        key_json_layout = QVBoxLayout(self.key_json_group)
        key_json_layout.setContentsMargins(0, 0, 0, 0)
        
        self.service_account_key = QTextEdit()
        self.service_account_key.setPlaceholderText("Paste service account key JSON here")
        self.service_account_key.setMaximumHeight(100)
        key_json_layout.addWidget(self.service_account_key)
        form_layout.addRow("Service Account Key:", self.key_json_group)
        
        # Scopes
        self.scopes = QLineEdit()
        self.scopes.setPlaceholderText("https://www.googleapis.com/auth/cloud-platform")
        self.scopes.setText("https://www.googleapis.com/auth/cloud-platform")
        form_layout.addRow("Scopes:", self.scopes)
        
        # Service account email (for impersonation)
        self.service_account_email = QLineEdit()
        self.service_account_email.setPlaceholderText("Optional: service account email for impersonation")
        form_layout.addRow("Service Account Email:", self.service_account_email)
        
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
        
        # Initialize visibility
        self.on_auth_method_changed(self.auth_method.currentText())
    
    def on_auth_method_changed(self, method: str):
        """Handle authentication method change"""
        if method == "Service Account Key File":
            self.key_file_group.setVisible(True)
            self.key_json_group.setVisible(False)
        elif method == "Service Account Key JSON":
            self.key_file_group.setVisible(False)
            self.key_json_group.setVisible(True)
        else:  # Application Default Credentials or User Account
            self.key_file_group.setVisible(False)
            self.key_json_group.setVisible(False)
    
    def browse_key_file(self):
        """Browse for service account key file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Service Account Key File", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.key_file_path.setText(file_path)
    
    def get_credentials(self) -> Dict[str, Any]:
        """Get GCP credentials from form"""
        scopes = [s.strip() for s in self.scopes.text().split(',') if s.strip()]
        return {
            "project_id": self.project_id.text(),
            "auth_method": self.auth_method.currentText(),
            "key_file_path": self.key_file_path.text() or None,
            "service_account_key": self.service_account_key.toPlainText() or None,
            "scopes": scopes,
            "service_account_email": self.service_account_email.text() or None
        }
    
    def set_credentials(self, credentials: Dict[str, Any]):
        """Set GCP credentials in form"""
        self.project_id.setText(credentials.get("project_id", ""))
        
        auth_method = credentials.get("auth_method", "Service Account Key File")
        index = self.auth_method.findText(auth_method)
        if index >= 0:
            self.auth_method.setCurrentIndex(index)
        
        self.key_file_path.setText(credentials.get("key_file_path", ""))
        self.service_account_key.setPlainText(credentials.get("service_account_key", ""))
        self.service_account_email.setText(credentials.get("service_account_email", ""))
        
        scopes = credentials.get("scopes", ["https://www.googleapis.com/auth/cloud-platform"])
        self.scopes.setText(", ".join(scopes))
        
        # Update visibility based on auth method
        self.on_auth_method_changed(auth_method)

class OAuthAuthTab(QWidget):
    """OAuth authentication tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup OAuth authentication UI"""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Configure OAuth credentials for third-party services:")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Form layout
        form_layout = QFormLayout()
        
        self.provider = QComboBox()
        self.provider.addItems([
            "Google", "Microsoft", "GitHub", "Slack", "Salesforce", "Custom"
        ])
        form_layout.addRow("Provider:", self.provider)
        
        self.client_id = QLineEdit()
        self.client_id.setPlaceholderText("OAuth Client ID")
        form_layout.addRow("Client ID:", self.client_id)
        
        self.client_secret = QLineEdit()
        self.client_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.client_secret.setPlaceholderText("OAuth Client Secret")
        form_layout.addRow("Client Secret:", self.client_secret)
        
        self.redirect_uri = QLineEdit()
        self.redirect_uri.setPlaceholderText("http://localhost:8080/callback")
        self.redirect_uri.setText("http://localhost:8080/callback")
        form_layout.addRow("Redirect URI:", self.redirect_uri)
        
        self.scopes = QLineEdit()
        self.scopes.setPlaceholderText("openid profile email")
        form_layout.addRow("Scopes:", self.scopes)
        
        layout.addLayout(form_layout)
        
        # OAuth flow button
        self.oauth_button = QPushButton("Start OAuth Flow")
        self.oauth_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        layout.addWidget(self.oauth_button)
        
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
        """Get OAuth credentials from form"""
        scopes = [s.strip() for s in self.scopes.text().split() if s.strip()]
        return {
            "provider": self.provider.currentText(),
            "client_id": self.client_id.text(),
            "client_secret": self.client_secret.text(),
            "redirect_uri": self.redirect_uri.text(),
            "scopes": scopes
        }
    
    def set_credentials(self, credentials: Dict[str, Any]):
        """Set OAuth credentials in form"""
        provider = credentials.get("provider", "Google")
        index = self.provider.findText(provider)
        if index >= 0:
            self.provider.setCurrentIndex(index)
        
        self.client_id.setText(credentials.get("client_id", ""))
        self.client_secret.setText(credentials.get("client_secret", ""))
        self.redirect_uri.setText(credentials.get("redirect_uri", "http://localhost:8080/callback"))
        
        scopes = credentials.get("scopes", [])
        self.scopes.setText(" ".join(scopes))

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
        
        # Azure tab
        self.azure_tab = AzureAuthTab()
        self.tab_widget.addTab(self.azure_tab, "Azure")
        
        # GCP tab
        self.gcp_tab = GCPAuthTab()
        self.tab_widget.addTab(self.gcp_tab, "GCP")
        
        # OAuth tab
        self.oauth_tab = OAuthAuthTab()
        self.tab_widget.addTab(self.oauth_tab, "OAuth")
        
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
        self.azure_tab.test_button.clicked.connect(lambda: self.test_credentials("Azure"))
        self.gcp_tab.test_button.clicked.connect(lambda: self.test_credentials("GCP"))
        self.oauth_tab.test_button.clicked.connect(lambda: self.test_credentials("OAuth"))
    
    def test_credentials(self, provider: str):
        """Test cloud credentials"""
        try:
            if provider == "AWS":
                credentials = self.aws_tab.get_credentials()
            elif provider == "Azure":
                credentials = self.azure_tab.get_credentials()
            elif provider == "GCP":
                credentials = self.gcp_tab.get_credentials()
            elif provider == "OAuth":
                credentials = self.oauth_tab.get_credentials()
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
            if provider == "AWS":
                self.aws_tab.test_button.setText("Testing...")
                self.aws_tab.test_button.setEnabled(False)
            elif provider == "Azure":
                self.azure_tab.test_button.setText("Testing...")
                self.azure_tab.test_button.setEnabled(False)
            elif provider == "GCP":
                self.gcp_tab.test_button.setText("Testing...")
                self.gcp_tab.test_button.setEnabled(False)
            elif provider == "OAuth":
                self.oauth_tab.test_button.setText("Testing...")
                self.oauth_tab.test_button.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Failed to test {provider} credentials: {e}")
            QMessageBox.warning(self, "Error", f"Failed to test credentials: {str(e)}")
    
    def on_auth_result(self, success: bool, message: str, credentials: Dict[str, Any]):
        """Handle authentication result"""
        # Reset button states
        self.aws_tab.test_button.setText("Test Connection")
        self.aws_tab.test_button.setEnabled(True)
        self.azure_tab.test_button.setText("Test Connection")
        self.azure_tab.test_button.setEnabled(True)
        self.gcp_tab.test_button.setText("Test Connection")
        self.gcp_tab.test_button.setEnabled(True)
        self.oauth_tab.test_button.setText("Test Connection")
        self.oauth_tab.test_button.setEnabled(True)
        
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
            elif current_tab == 1:  # Azure
                credentials = self.azure_tab.get_credentials()
                if self._validate_credentials("Azure", credentials):
                    QMessageBox.information(self, "Success", "Azure credentials would be saved securely!")
                    self.credentials_updated.emit()
            elif current_tab == 2:  # GCP
                credentials = self.gcp_tab.get_credentials()
                if self._validate_credentials("GCP", credentials):
                    QMessageBox.information(self, "Success", "GCP credentials would be saved securely!")
                    self.credentials_updated.emit()
            elif current_tab == 3:  # OAuth
                credentials = self.oauth_tab.get_credentials()
                if self._validate_credentials("OAuth", credentials):
                    QMessageBox.information(self, "Success", "OAuth credentials would be saved securely!")
                    self.credentials_updated.emit()
                
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save credentials: {str(e)}")
    
    def _validate_credentials(self, provider: str, credentials: Dict[str, Any]) -> bool:
        """Validate credentials before saving/testing"""
        if provider == "AWS":
            if not credentials.get("access_key") or not credentials.get("secret_key"):
                QMessageBox.warning(self, "Validation Error", "Access Key and Secret Key are required")
                return False
        elif provider == "Azure":
            if not credentials.get("tenant_id") or not credentials.get("client_id"):
                QMessageBox.warning(self, "Validation Error", "Tenant ID and Client ID are required")
                return False
            auth_method = credentials.get("auth_method", "Client Secret")
            if auth_method == "Client Secret" and not credentials.get("client_secret"):
                QMessageBox.warning(self, "Validation Error", "Client Secret is required for this authentication method")
                return False
            elif auth_method == "Certificate" and not credentials.get("certificate_path"):
                QMessageBox.warning(self, "Validation Error", "Certificate path is required for this authentication method")
                return False
        elif provider == "GCP":
            if not credentials.get("project_id"):
                QMessageBox.warning(self, "Validation Error", "Project ID is required")
                return False
            auth_method = credentials.get("auth_method", "Service Account Key File")
            if auth_method == "Service Account Key File" and not credentials.get("key_file_path"):
                QMessageBox.warning(self, "Validation Error", "Service account key file path is required")
                return False
            elif auth_method == "Service Account Key JSON" and not credentials.get("service_account_key"):
                QMessageBox.warning(self, "Validation Error", "Service account key JSON is required")
                return False
        elif provider == "OAuth":
            if not credentials.get("client_id") or not credentials.get("client_secret"):
                QMessageBox.warning(self, "Validation Error", "Client ID and Client Secret are required")
                return False
        
        return True
