"""
Login Dialog
Authentication interface for the Data Platform application
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, 
    QPushButton, QLabel, QCheckBox, QComboBox, QFrame, QMessageBox,
    QProgressBar, QTextEdit, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon, QKeySequence, QShortcut
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AuthenticationWorker(QThread):
    """Background worker for authentication operations"""
    
    auth_result = pyqtSignal(bool, str)  # success, message
    progress_updated = pyqtSignal(int)  # progress percentage
    
    def __init__(self, auth_type: str, credentials: Dict[str, Any]):
        super().__init__()
        self.auth_type = auth_type
        self.credentials = credentials
        
    def run(self):
        """Perform authentication in background"""
        try:
            self.progress_updated.emit(10)
            
            if self.auth_type == "local":
                success, message = self.authenticate_local()
            elif self.auth_type == "active_directory":
                success, message = self.authenticate_ad()
            elif self.auth_type == "oauth":
                success, message = self.authenticate_oauth()
            else:
                success, message = False, "Unknown authentication type"
                
            self.progress_updated.emit(100)
            self.auth_result.emit(success, message)
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self.auth_result.emit(False, f"Authentication failed: {str(e)}")
            
    def authenticate_local(self) -> tuple[bool, str]:
        """Authenticate using local credentials"""
        import time
        time.sleep(1)  # Simulate authentication delay
        
        username = self.credentials.get('username', '')
        password = self.credentials.get('password', '')
        
        # Check for stored credentials (simplified implementation)
        credentials_file = Path.home() / ".dataplatform" / "credentials.json"
        
        if credentials_file.exists():
            try:
                with open(credentials_file, 'r') as f:
                    stored_creds = json.load(f)
                    
                # Simple hash-based verification (in production, use proper password hashing)
                stored_hash = stored_creds.get('password_hash', '')
                provided_hash = hashlib.sha256(password.encode()).hexdigest()
                
                if stored_creds.get('username') == username and stored_hash == provided_hash:
                    return True, "Local authentication successful"
                else:
                    return False, "Invalid username or password"
                    
            except Exception as e:
                logger.error(f"Error reading credentials: {e}")
                return False, "Error reading stored credentials"
        else:
            # First time setup - create new credentials
            if username and password:
                return self.create_local_account(username, password)
            else:
                return False, "Please provide username and password"
                
    def create_local_account(self, username: str, password: str) -> tuple[bool, str]:
        """Create new local account"""
        try:
            credentials_dir = Path.home() / ".dataplatform"
            credentials_dir.mkdir(parents=True, exist_ok=True)
            
            credentials_file = credentials_dir / "credentials.json"
            
            # Hash password (in production, use bcrypt or similar)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            credentials = {
                'username': username,
                'password_hash': password_hash,
                'created': str(Path.cwd()),  # Use current timestamp in production
                'auth_type': 'local'
            }
            
            with open(credentials_file, 'w') as f:
                json.dump(credentials, f)
                
            return True, "Local account created successfully"
            
        except Exception as e:
            logger.error(f"Error creating local account: {e}")
            return False, f"Failed to create account: {str(e)}"
            
    def authenticate_ad(self) -> tuple[bool, str]:
        """Authenticate using Active Directory"""
        # Placeholder for AD authentication
        import time
        time.sleep(2)  # Simulate AD authentication delay
        
        # In production, this would use libraries like python-ldap or adal
        return False, "Active Directory authentication not yet implemented"
        
    def authenticate_oauth(self) -> tuple[bool, str]:
        """Authenticate using OAuth (Google, Microsoft, etc.)"""
        # Placeholder for OAuth authentication
        import time
        time.sleep(1.5)  # Simulate OAuth flow
        
        # In production, this would use OAuth libraries
        return False, "OAuth authentication not yet implemented"

class LoginDialog(QDialog):
    """Main login dialog for the application"""
    
    authentication_successful = pyqtSignal(dict)  # user_info
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_info = {}
        self.auth_worker = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the login dialog UI"""
        self.setWindowTitle("Data Platform - Login")
        self.setModal(True)
        self.setFixedSize(450, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            QFrame#loginFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add spacing
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Main login frame
        login_frame = QFrame()
        login_frame.setObjectName("loginFrame")
        frame_layout = QVBoxLayout(login_frame)
        frame_layout.setContentsMargins(30, 30, 30, 30)
        frame_layout.setSpacing(20)
        
        # Application logo/title
        title_layout = QVBoxLayout()
        title_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        app_title = QLabel("Data Platform")
        app_title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_title.setStyleSheet("color: #007bff; margin-bottom: 5px;")
        title_layout.addWidget(app_title)
        
        subtitle = QLabel("Unified Data Management & Analytics")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #6c757d; margin-bottom: 20px;")
        title_layout.addWidget(subtitle)
        
        frame_layout.addLayout(title_layout)
        
        # Authentication type selection
        auth_group = QGroupBox("Authentication Method")
        auth_layout = QVBoxLayout(auth_group)
        
        self.auth_type = QComboBox()
        self.auth_type.addItems([
            "Local Account",
            "Active Directory", 
            "OAuth (Google/Microsoft)"
        ])
        self.auth_type.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background: white;
            }
        """)
        auth_layout.addWidget(self.auth_type)
        
        frame_layout.addWidget(auth_group)
        
        # Credentials form
        self.credentials_form = QFormLayout()
        
        # Username
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background: white;
                font-size: 11pt;
            }
            QLineEdit:focus {
                border-color: #007bff;
            }
        """)
        self.credentials_form.addRow("Username:", self.username_input)
        
        # Password
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setStyleSheet(self.username_input.styleSheet())
        self.credentials_form.addRow("Password:", self.password_input)
        
        frame_layout.addLayout(self.credentials_form)
        
        # Additional options
        options_layout = QHBoxLayout()
        
        self.remember_me = QCheckBox("Remember me")
        self.remember_me.setStyleSheet("QCheckBox { color: #495057; }")
        options_layout.addWidget(self.remember_me)
        
        options_layout.addStretch()
        
        forgot_password = QLabel('<a href="#" style="color: #007bff; text-decoration: none;">Forgot password?</a>')
        forgot_password.mousePressEvent = self.show_forgot_password
        options_layout.addWidget(forgot_password)
        
        frame_layout.addLayout(options_layout)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
        """)
        frame_layout.addWidget(self.progress_bar)
        
        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        frame_layout.addWidget(self.login_button)
        
        # Create account link
        create_account_layout = QHBoxLayout()
        create_account_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        create_account_label = QLabel("Don't have an account?")
        create_account_label.setStyleSheet("color: #6c757d;")
        create_account_layout.addWidget(create_account_label)
        
        create_account_link = QLabel('<a href="#" style="color: #007bff; text-decoration: none; margin-left: 5px;">Create one</a>')
        create_account_link.mousePressEvent = self.show_create_account
        create_account_layout.addWidget(create_account_link)
        
        frame_layout.addLayout(create_account_layout)
        
        layout.addWidget(login_frame)
        
        # Add spacing
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Status message area
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setVisible(False)
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.login_button.clicked.connect(self.attempt_login)
        self.password_input.returnPressed.connect(self.attempt_login)
        self.auth_type.currentTextChanged.connect(self.on_auth_type_changed)
        
        # Setup keyboard shortcuts
        login_shortcut = QShortcut(QKeySequence("Return"), self)
        login_shortcut.activated.connect(self.attempt_login)
        
    def on_auth_type_changed(self, auth_type: str):
        """Handle authentication type changes"""
        if auth_type == "Local Account":
            self.username_input.setVisible(True)
            self.password_input.setVisible(True)
            self.credentials_form.labelForField(self.username_input).setVisible(True)
            self.credentials_form.labelForField(self.password_input).setVisible(True)
        elif auth_type == "Active Directory":
            self.username_input.setPlaceholderText("domain\\username")
            self.username_input.setVisible(True)
            self.password_input.setVisible(True)
        elif auth_type == "OAuth (Google/Microsoft)":
            self.username_input.setVisible(False)
            self.password_input.setVisible(False)
            self.credentials_form.labelForField(self.username_input).setVisible(False)
            self.credentials_form.labelForField(self.password_input).setVisible(False)
            
    def attempt_login(self):
        """Attempt to authenticate user"""
        auth_type_text = self.auth_type.currentText()
        
        if auth_type_text == "Local Account":
            auth_type = "local"
        elif auth_type_text == "Active Directory":
            auth_type = "active_directory"
        else:
            auth_type = "oauth"
            
        credentials = {
            'username': self.username_input.text(),
            'password': self.password_input.text(),
            'remember_me': self.remember_me.isChecked()
        }
        
        # Validate input
        if auth_type in ["local", "active_directory"]:
            if not credentials['username'] or not credentials['password']:
                self.show_status_message("Please enter both username and password", "error")
                return
                
        # Disable login button and show progress
        self.login_button.setEnabled(False)
        self.login_button.setText("Authenticating...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start authentication worker
        self.auth_worker = AuthenticationWorker(auth_type, credentials)
        self.auth_worker.auth_result.connect(self.on_auth_result)
        self.auth_worker.progress_updated.connect(self.progress_bar.setValue)
        self.auth_worker.start()
        
    def on_auth_result(self, success: bool, message: str):
        """Handle authentication result"""
        # Reset UI
        self.login_button.setEnabled(True)
        self.login_button.setText("Login")
        self.progress_bar.setVisible(False)
        
        if success:
            self.show_status_message(message, "success")
            
            # Store user info
            self.user_info = {
                'username': self.username_input.text(),
                'auth_type': self.auth_type.currentText(),
                'login_time': str(Path.cwd()),  # Use actual timestamp in production
                'remember_me': self.remember_me.isChecked()
            }
            
            # Close dialog after short delay
            QTimer.singleShot(1000, self.accept_login)
            
        else:
            self.show_status_message(message, "error")
            
    def accept_login(self):
        """Accept successful login"""
        self.authentication_successful.emit(self.user_info)
        self.accept()
        
    def show_status_message(self, message: str, message_type: str):
        """Show status message with appropriate styling"""
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        
        if message_type == "error":
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
        elif message_type == "success":
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
            
        # Auto-hide error messages
        if message_type == "error":
            QTimer.singleShot(5000, lambda: self.status_label.setVisible(False))
            
    def show_forgot_password(self, event):
        """Show forgot password dialog"""
        QMessageBox.information(
            self, 
            "Forgot Password", 
            "Password recovery is not yet implemented.\n\nFor local accounts, you can reset by deleting:\n~/.dataplatform/credentials.json"
        )
        
    def show_create_account(self, event):
        """Show create account information"""
        QMessageBox.information(
            self, 
            "Create Account", 
            "To create a local account, simply enter a new username and password and click Login.\n\nFor Active Directory or OAuth, contact your system administrator."
        )
        
    def get_user_info(self) -> Dict[str, Any]:
        """Get authenticated user information"""
        return self.user_info
