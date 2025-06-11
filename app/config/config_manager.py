"""
Configuration Manager
Handles application configuration storage and retrieval in JSON format
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration in JSON format"""
    
    def __init__(self):
        self.config_path: Optional[Path] = None
        self.config_data: Dict[str, Any] = self.get_default_config()
        self.temp_config_path = Path.home() / ".dataplatform" / "temp_config.json"
        self.is_temporary = True
        
        # Ensure temp directory exists
        self.temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing configuration
        self.load_or_prompt_config()
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure"""
        return {
            "application": {
                "name": "Untitled Application",
                "version": "1.0.0",
                "description": "",
                "created_date": "",
                "last_modified": ""
            },
            "databases": {
                "connections": [],
                "default_backend": "sqlite"
            },
            "ai_tools": {
                "anthropic": {
                    "enabled": False,
                    "api_key": "",
                    "model": "claude-3-sonnet-20240229"
                },
                "openai": {
                    "enabled": False,
                    "api_key": "",
                    "model": "gpt-4"
                },
                "local_models": {
                    "enabled": False,
                    "models": ["qwen2.5:7b", "gemma2:7b"],
                    "default_model": "qwen2.5:7b"
                }
            },
            "vector_search": {
                "database_type": "internal",
                "database_path": "",
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "file_management": {
                "watched_directories": [],
                "supported_formats": [".txt", ".pdf", ".docx", ".csv", ".json", ".xlsx"],
                "index_on_startup": True
            },
            "datasets": {
                "registered_datasets": [],
                "access_permissions": {},
                "metadata": {}
            },
            "governance": {
                "documentation": {},
                "permissions": {},
                "tags": [],
                "metadata_schema": {}
            },
            "workflows": {
                "tasks": [],
                "schedules": [],
                "agents": []
            },
            "security": {
                "encryption_enabled": True,
                "password_manager": "system",
                "ssl_verification": True
            },
            "vector_search": {
                "database_path": "",
                "embeddings_path": "",
                "model": "all-MiniLM-L6-v2",
                "max_chunk_size": 512,
                "chunk_overlap": 50,
                "similarity_threshold": 0.3,
                "max_results": 25
            },
            "ui_preferences": {
                "theme": "system",
                "layout": "default",
                "recent_searches": []
            }
        }
        
    def load_or_prompt_config(self):
        """Load existing config or prompt user for config path"""
        # Check if temp config exists
        if self.temp_config_path.exists():
            try:
                self.load_config(self.temp_config_path)
                self.is_temporary = True
                logger.info("Loaded temporary configuration")
                return
            except Exception as e:
                logger.warning(f"Failed to load temporary config: {e}")
        
        # Check for config in current directory
        current_dir_config = Path.cwd() / "dataplatform_config.json"
        if current_dir_config.exists():
            try:
                self.load_config(current_dir_config)
                self.is_temporary = False
                logger.info(f"Loaded configuration from {current_dir_config}")
                return
            except Exception as e:
                logger.warning(f"Failed to load config from current directory: {e}")
                
        # Save default config to temp location
        self.save_temp_config()
        
    def load_config(self, config_path: Path):
        """Load configuration from specified path"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # Merge with default config to ensure all keys exist
            self.config_data = self.merge_config(self.get_default_config(), loaded_config)
            self.config_path = config_path
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
            
    def merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with default config"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_config(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def save_config(self, config_path: Optional[Path] = None):
        """Save configuration to specified path or current config path"""
        if config_path is None:
            config_path = self.config_path
            
        if config_path is None:
            raise ValueError("No configuration path specified")
            
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                
            self.config_path = config_path
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
            
    def save_temp_config(self):
        """Save configuration to temporary location"""
        try:
            self.save_config(self.temp_config_path)
            self.is_temporary = True
        except Exception as e:
            logger.error(f"Failed to save temporary configuration: {e}")
            
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'ai_tools.anthropic.enabled')"""
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config_data
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        # Set the final value
        config[keys[-1]] = value
        
        # Auto-save to temp location
        if self.is_temporary:
            self.save_temp_config()
            
    def new_application(self):
        """Create new application configuration"""
        self.config_data = self.get_default_config()
        self.config_path = None
        self.is_temporary = True
        self.save_temp_config()
        
    def save_application(self):
        """Save application configuration to permanent location"""
        if self.is_temporary or self.config_path is None:
            # Prompt for save location
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Application Configuration",
                str(Path.home() / "dataplatform_config.json"),
                "JSON Files (*.json);;All Files (*)"
            )
            
            if file_path:
                self.save_config(Path(file_path))
                self.is_temporary = False
                
                # Clean up temp config
                if self.temp_config_path.exists():
                    try:
                        self.temp_config_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary config: {e}")
                        
        else:
            # Save to existing location
            self.save_config()
            
    def add_database_connection(self, connection_config: Dict[str, Any]):
        """Add database connection configuration"""
        connections = self.get("databases.connections", [])
        connections.append(connection_config)
        self.set("databases.connections", connections)
        
    def add_dataset(self, dataset_config: Dict[str, Any]):
        """Add dataset configuration"""
        datasets = self.get("datasets.registered_datasets", [])
        datasets.append(dataset_config)
        self.set("datasets.registered_datasets", datasets)
        
    def add_watched_directory(self, directory_path: str):
        """Add directory to watched directories for file indexing"""
        watched_dirs = self.get("file_management.watched_directories", [])
        if directory_path not in watched_dirs:
            watched_dirs.append(directory_path)
            self.set("file_management.watched_directories", watched_dirs)
            
    def get_ai_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get AI tool configuration"""
        return self.get(f"ai_tools.{tool_name}", {})
        
    def is_ai_tool_enabled(self, tool_name: str) -> bool:
        """Check if AI tool is enabled"""
        return self.get(f"ai_tools.{tool_name}.enabled", False)
        
    def get_database_connections(self) -> list:
        """Get all database connections"""
        return self.get("databases.connections", [])
        
    def get_registered_datasets(self) -> list:
        """Get all registered datasets"""
        return self.get("datasets.registered_datasets", [])
