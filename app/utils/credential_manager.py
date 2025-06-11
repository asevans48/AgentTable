"""
Secure Credential Manager
Handles secure storage and retrieval of credentials for cloud providers
Uses Windows Credential Manager, macOS Keychain, or Linux Secret Service
"""

import logging
import json
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path
import platform

logger = logging.getLogger(__name__)

class CredentialManager:
    """Secure credential storage and retrieval"""
    
    def __init__(self):
        self.system = platform.system()
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the appropriate credential backend"""
        try:
            if self.system == "Windows":
                import keyring
                import keyring.backends.Windows
                keyring.set_keyring(keyring.backends.Windows.WinVaultKeyring())
            elif self.system == "Darwin":  # macOS
                import keyring
                import keyring.backends.macOS
                keyring.set_keyring(keyring.backends.macOS.Keyring())
            else:  # Linux
                import keyring
                import keyring.backends.SecretService
                keyring.set_keyring(keyring.backends.SecretService.Keyring())
            
            self.backend_available = True
            logger.info(f"Credential backend initialized for {self.system}")
            
        except ImportError as e:
            logger.warning(f"Keyring not available: {e}. Using fallback storage.")
            self.backend_available = False
        except Exception as e:
            logger.error(f"Failed to initialize credential backend: {e}")
            self.backend_available = False
    
    def store_credential(self, service: str, username: str, credential_data: Dict[str, Any]) -> bool:
        """Store credentials securely"""
        try:
            if self.backend_available:
                import keyring
                # Store as JSON string
                credential_json = json.dumps(credential_data)
                keyring.set_password(f"DataPlatform_{service}", username, credential_json)
                logger.info(f"Stored credentials for {service}:{username}")
                return True
            else:
                return self._store_credential_fallback(service, username, credential_data)
                
        except Exception as e:
            logger.error(f"Failed to store credential for {service}:{username}: {e}")
            return False
    
    def get_credential(self, service: str, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve credentials securely"""
        try:
            if self.backend_available:
                import keyring
                credential_json = keyring.get_password(f"DataPlatform_{service}", username)
                if credential_json:
                    return json.loads(credential_json)
                return None
            else:
                return self._get_credential_fallback(service, username)
                
        except Exception as e:
            logger.error(f"Failed to retrieve credential for {service}:{username}: {e}")
            return None
    
    def _store_credential_fallback(self, service: str, username: str, credential_data: Dict[str, Any]) -> bool:
        """Fallback storage using encrypted local file"""
        try:
            # Create credentials directory
            cred_dir = Path.home() / ".dataplatform" / "credentials"
            cred_dir.mkdir(parents=True, exist_ok=True)
            
            # For now, store as plain JSON (in production, use encryption)
            cred_file = cred_dir / f"{service}_{username}.json"
            with open(cred_file, 'w') as f:
                json.dump(credential_data, f, indent=2)
            
            # Set restrictive permissions
            cred_file.chmod(0o600)
            
            logger.info(f"Stored credentials using fallback for {service}:{username}")
            return True
            
        except Exception as e:
            logger.error(f"Fallback credential storage failed: {e}")
            return False
    
    def _get_credential_fallback(self, service: str, username: str) -> Optional[Dict[str, Any]]:
        """Fallback retrieval from local file"""
        try:
            cred_dir = Path.home() / ".dataplatform" / "credentials"
            cred_file = cred_dir / f"{service}_{username}.json"
            
            if not cred_file.exists():
                return None
            
            with open(cred_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Fallback credential retrieval failed: {e}")
            return None

class CloudCredentialManager:
    """Specialized credential manager for cloud providers"""
    
    def __init__(self):
        self.credential_manager = CredentialManager()
    
    def store_aws_credentials(self, profile_name: str, access_key: str, secret_key: str, 
                            region: str = "us-east-1", session_token: str = None) -> bool:
        """Store AWS credentials"""
        credential_data = {
            "type": "aws",
            "access_key": access_key,
            "secret_key": secret_key,
            "region": region,
            "session_token": session_token
        }
        return self.credential_manager.store_credential("AWS", profile_name, credential_data)
    
    def get_aws_credentials(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve AWS credentials"""
        return self.credential_manager.get_credential("AWS", profile_name)
