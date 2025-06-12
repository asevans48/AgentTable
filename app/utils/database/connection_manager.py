"""
Database Connection Manager
Handles connections to various database types with credential management
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """Manages database connections across different database types"""
    
    def __init__(self, config_manager, credential_manager=None):
        self.config_manager = config_manager
        self.credential_manager = credential_manager
        self._connections = {}
        
    def get_connection(self, connection_name: str):
        """Get a database connection by name"""
        if connection_name in self._connections:
            return self._connections[connection_name]
            
        # Get connection config
        connections = self.config_manager.get("database.connections", {})
        if connection_name not in connections:
            raise ValueError(f"Connection '{connection_name}' not found in configuration")
            
        conn_config = connections[connection_name]
        db_type = conn_config.get('type', '').lower()
        
        try:
            if db_type == 'postgresql':
                connection = self._create_postgresql_connection(conn_config)
            elif db_type in ['mssql', 'sqlserver', 'azure_sql']:
                connection = self._create_mssql_connection(conn_config)
            elif db_type == 'sqlite':
                connection = self._create_sqlite_connection(conn_config)
            elif db_type == 'duckdb':
                connection = self._create_duckdb_connection(conn_config)
            elif db_type == 'bigquery':
                connection = self._create_bigquery_connection(conn_config)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
            self._connections[connection_name] = connection
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection '{connection_name}': {e}")
            raise
    
    def _create_postgresql_connection(self, config: Dict[str, Any]):
        """Create PostgreSQL connection"""
        try:
            import psycopg2
            
            # Get credentials
            password = self._get_password(config)
            
            conn_params = {
                'host': config['host'],
                'port': config.get('port', 5432),
                'database': config['database'],
                'user': config['username']
            }
            
            if password:
                conn_params['password'] = password
                
            # SSL configuration
            if config.get('ssl_disabled', False):
                conn_params['sslmode'] = 'disable'
            elif config.get('ssl_verify', True):
                conn_params['sslmode'] = 'require'
            else:
                conn_params['sslmode'] = 'prefer'
                
            return psycopg2.connect(**conn_params)
            
        except ImportError:
            raise ImportError("PostgreSQL driver not available. Install with: pip install psycopg2-binary")
    
    def _create_mssql_connection(self, config: Dict[str, Any]):
        """Create MSSQL/SQL Server connection"""
        try:
            import pyodbc
            
            # Build connection string
            conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={config['host']}"
            
            if config.get('port'):
                conn_str += f",{config['port']}"
                
            conn_str += f";DATABASE={config['database']}"
            
            # Authentication
            if config.get('use_ad_auth', False):
                conn_str += ";Trusted_Connection=yes"
            else:
                password = self._get_password(config)
                conn_str += f";UID={config['username']};PWD={password or ''}"
            
            # SSL configuration
            if config.get('ssl_disabled', False):
                conn_str += ";Encrypt=no"
            elif not config.get('ssl_verify', True):
                conn_str += ";Encrypt=yes;TrustServerCertificate=yes"
            else:
                conn_str += ";Encrypt=yes"
                
            return pyodbc.connect(conn_str)
            
        except ImportError:
            raise ImportError("SQL Server driver not available. Install with: pip install pyodbc")
    
    def _create_sqlite_connection(self, config: Dict[str, Any]):
        """Create SQLite connection"""
        import sqlite3
        
        db_path = config.get('path', config.get('database', 'data/local.db'))
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        return sqlite3.connect(db_path)
    
    def _create_duckdb_connection(self, config: Dict[str, Any]):
        """Create DuckDB connection"""
        try:
            import duckdb
            
            db_path = config.get('path', config.get('database', 'data/local.duckdb'))
            
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            return duckdb.connect(db_path)
            
        except ImportError:
            raise ImportError("DuckDB not available. Install with: pip install duckdb")
    
    def _create_bigquery_connection(self, config: Dict[str, Any]):
        """Create BigQuery connection"""
        try:
            from google.cloud import bigquery
            
            if config.get('service_account_path'):
                return bigquery.Client.from_service_account_json(config['service_account_path'])
            else:
                return bigquery.Client(project=config['project_id'])
                
        except ImportError:
            raise ImportError("BigQuery client not available. Install with: pip install google-cloud-bigquery")
    
    def _get_password(self, config: Dict[str, Any]) -> Optional[str]:
        """Get password from credential manager or config"""
        if self.credential_manager and config.get('use_credential_manager', True):
            try:
                service = f"database_{config.get('host', 'local')}"
                username = config.get('username', 'default')
                creds = self.credential_manager.get_credential(service, username)
                if creds:
                    return creds.get('password')
            except Exception as e:
                logger.warning(f"Failed to get password from credential manager: {e}")
        
        return config.get('password')
    
    def test_connection(self, connection_name: str) -> Dict[str, Any]:
        """Test a database connection"""
        try:
            conn = self.get_connection(connection_name)
            
            # Simple test query based on database type
            connections = self.config_manager.get("database.connections", {})
            db_type = connections[connection_name].get('type', '').lower()
            
            if db_type == 'postgresql':
                cursor = conn.cursor()
                cursor.execute("SELECT version()")
                result = cursor.fetchone()[0]
                cursor.close()
            elif db_type in ['mssql', 'sqlserver', 'azure_sql']:
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION")
                result = cursor.fetchone()[0]
                cursor.close()
            elif db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute("SELECT sqlite_version()")
                result = cursor.fetchone()[0]
                cursor.close()
            elif db_type == 'duckdb':
                result = conn.execute("SELECT version()").fetchone()[0]
            elif db_type == 'bigquery':
                # For BigQuery, just try to list datasets
                datasets = list(conn.list_datasets())
                result = f"Connected to BigQuery with {len(datasets)} datasets"
            else:
                result = "Connected successfully"
            
            return {
                'success': True,
                'message': 'Connection successful',
                'details': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}',
                'details': None
            }
        finally:
            # Close connection if it was created for testing
            if connection_name in self._connections:
                try:
                    self._connections[connection_name].close()
                    del self._connections[connection_name]
                except:
                    pass
    
    def close_all_connections(self):
        """Close all open connections"""
        for name, conn in self._connections.items():
            try:
                conn.close()
                logger.info(f"Closed connection: {name}")
            except Exception as e:
                logger.warning(f"Error closing connection {name}: {e}")
        
        self._connections.clear()
    
    def get_available_connections(self) -> List[str]:
        """Get list of available connection names"""
        connections = self.config_manager.get("database.connections", {})
        return [name for name, config in connections.items() if config.get('enabled', True)]
