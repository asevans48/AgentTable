"""
AI Chat Service
Handles AI conversations with context from selected files and datasets
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class AIContextBuilder:
    """Builds context from selected items for AI chat"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.max_context_tokens = 8000  # Conservative limit for context
        self.max_sample_rows = 10  # Maximum rows to include in data samples
        
    def build_context(self, selected_items: List[Dict[str, Any]], user_question: str) -> Dict[str, Any]:
        """Build comprehensive context from selected items"""
        context = {
            'user_question': user_question,
            'selected_items_count': len(selected_items),
            'files': [],
            'datasets': [],
            'metadata_summary': '',
            'data_samples': [],
            'schema_info': [],
            'total_tokens_estimate': 0
        }
        
        # Separate files and datasets
        files = [item for item in selected_items if item['type'] == 'file']
        datasets = [item for item in selected_items if item['type'] == 'dataset']
        
        # Process files
        for file_item in files:
            file_context = self._process_file_item(file_item)
            if file_context:
                context['files'].append(file_context)
                
        # Process datasets
        for dataset_item in datasets:
            dataset_context = self._process_dataset_item(dataset_item)
            if dataset_context:
                context['datasets'].append(dataset_context)
                
        # Build metadata summary
        context['metadata_summary'] = self._build_metadata_summary(context)
        
        # Estimate token usage
        context['total_tokens_estimate'] = self._estimate_tokens(context)
        
        return context
    
    def _process_file_item(self, file_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a file item to extract relevant context"""
        try:
            result_data = file_item.get('result_data', {})
            file_path = file_item.get('path', '')
            
            file_context = {
                'name': file_item.get('name', 'Unknown'),
                'path': file_path,
                'type': result_data.get('file_type', 'Unknown'),
                'summary': result_data.get('summary', ''),
                'content_preview': '',
                'metadata': {
                    'size': result_data.get('file_size', 'Unknown'),
                    'last_modified': result_data.get('last_modified', 'Unknown'),
                    'owner': result_data.get('owner', 'Unknown')
                }
            }
            
            # Try to read file content for preview
            if file_path and Path(file_path).exists():
                try:
                    file_path_obj = Path(file_path)
                    if file_path_obj.suffix.lower() in ['.txt', '.md', '.py', '.sql', '.json', '.csv']:
                        with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(2000)  # Read first 2000 characters
                            file_context['content_preview'] = content
                            
                        # For CSV files, try to extract schema
                        if file_path_obj.suffix.lower() == '.csv':
                            schema = self._extract_csv_schema(file_path_obj)
                            if schema:
                                file_context['schema'] = schema
                                
                        # For JSON files, try to extract structure
                        elif file_path_obj.suffix.lower() == '.json':
                            structure = self._extract_json_structure(file_path_obj)
                            if structure:
                                file_context['structure'] = structure
                                
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
                    file_context['content_preview'] = f"Could not read file: {str(e)}"
            
            return file_context
            
        except Exception as e:
            logger.error(f"Error processing file item: {e}")
            return None
    
    def _process_dataset_item(self, dataset_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a dataset item to extract relevant context"""
        try:
            result_data = dataset_item.get('result_data', {})
            
            dataset_context = {
                'name': dataset_item.get('name', 'Unknown'),
                'type': result_data.get('file_type', 'Unknown'),
                'description': result_data.get('user_description', ''),
                'schema_info': result_data.get('schema_info', ''),
                'tags': result_data.get('tags', []),
                'metadata': {
                    'owner': result_data.get('owner', 'Unknown'),
                    'access_level': result_data.get('access_level', 'Unknown'),
                    'last_modified': result_data.get('last_modified', 'Unknown')
                },
                'data_sample': None
            }
            
            # Try to get data sample if it's a database dataset
            if dataset_context['type'].lower() in ['table', 'view']:
                sample_data = self._get_dataset_sample(dataset_item)
                if sample_data:
                    dataset_context['data_sample'] = sample_data
            
            return dataset_context
            
        except Exception as e:
            logger.error(f"Error processing dataset item: {e}")
            return None
    
    def _extract_csv_schema(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract schema information from CSV file"""
        try:
            import pandas as pd
            
            # Read just the first few rows to get schema
            df = pd.read_csv(file_path, nrows=5)
            
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
            }
            
            return schema
            
        except Exception as e:
            logger.warning(f"Could not extract CSV schema from {file_path}: {e}")
            return None
    
    def _extract_json_structure(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract structure information from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            structure = {
                'type': type(data).__name__,
                'sample_data': None
            }
            
            if isinstance(data, dict):
                structure['keys'] = list(data.keys())
                # Include a small sample
                sample = {k: v for k, v in list(data.items())[:5]}
                structure['sample_data'] = sample
            elif isinstance(data, list) and len(data) > 0:
                structure['length'] = len(data)
                if isinstance(data[0], dict):
                    structure['item_keys'] = list(data[0].keys())
                structure['sample_data'] = data[:3]  # First 3 items
            
            return structure
            
        except Exception as e:
            logger.warning(f"Could not extract JSON structure from {file_path}: {e}")
            return None
    
    def _get_dataset_sample(self, dataset_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get sample data from dataset if possible"""
        try:
            # This would connect to the database and get sample data
            # For now, return a placeholder
            result_data = dataset_item.get('result_data', {})
            
            # Check if we have SQL data from a previous query
            if 'sql_data' in result_data:
                return {
                    'type': 'sql_result',
                    'data': [result_data['sql_data']],
                    'note': 'Sample from previous SQL query'
                }
            
            # Try to get sample using database connection
            dataset_name = dataset_item.get('name', '')
            if dataset_name and result_data.get('file_type', '').lower() in ['table', 'view']:
                sample = self._query_dataset_sample(dataset_name)
                if sample:
                    return sample
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get dataset sample: {e}")
            return None
    
    def _query_dataset_sample(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Query database for dataset sample"""
        try:
            from utils.database.connection_manager import DatabaseConnectionManager
            
            # Get database connections
            db_connections = self.config_manager.get("database.connections", {})
            if not db_connections:
                return None
            
            # Use first available connection
            connection_name = list(db_connections.keys())[0]
            db_manager = DatabaseConnectionManager(self.config_manager)
            connection = db_manager.get_connection(connection_name)
            
            if not connection:
                return None
            
            # Query sample data
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM {dataset_name} LIMIT {self.max_sample_rows}")
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            if rows:
                sample_data = []
                for row in rows:
                    sample_data.append(dict(zip(columns, row)))
                
                return {
                    'type': 'database_sample',
                    'columns': columns,
                    'data': sample_data,
                    'row_count': len(rows),
                    'note': f'Sample of {len(rows)} rows from {dataset_name}'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not query dataset sample for {dataset_name}: {e}")
            return None
    
    def _build_metadata_summary(self, context: Dict[str, Any]) -> str:
        """Build a summary of all metadata"""
        summary_parts = []
        
        if context['files']:
            summary_parts.append(f"Files ({len(context['files'])}):")
            for file_ctx in context['files']:
                summary_parts.append(f"  • {file_ctx['name']} ({file_ctx['type']})")
        
        if context['datasets']:
            summary_parts.append(f"Datasets ({len(context['datasets'])}):")
            for dataset_ctx in context['datasets']:
                summary_parts.append(f"  • {dataset_ctx['name']} ({dataset_ctx['type']})")
        
        return '\n'.join(summary_parts)
    
    def _estimate_tokens(self, context: Dict[str, Any]) -> int:
        """Estimate token count for context"""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = 0
        
        # Count characters in all text content
        total_chars += len(context.get('user_question', ''))
        total_chars += len(context.get('metadata_summary', ''))
        
        for file_ctx in context.get('files', []):
            total_chars += len(str(file_ctx))
            
        for dataset_ctx in context.get('datasets', []):
            total_chars += len(str(dataset_ctx))
        
        return total_chars // 4  # Rough token estimate

class AIService:
    """Main AI service for handling chat requests"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.context_builder = AIContextBuilder(config_manager)
        
    def chat_with_context(self, user_question: str, selected_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process AI chat request with context"""
        try:
            # Build context from selected items
            context = self.context_builder.build_context(selected_items, user_question)
            
            # Check which AI services are available
            available_services = self._get_available_ai_services()
            
            if not available_services:
                return {
                    'success': False,
                    'error': 'No AI services configured. Please configure API keys in Settings.',
                    'context': context
                }
            
            # Try each available service
            for service_name in available_services:
                try:
                    response = self._call_ai_service(service_name, context)
                    if response['success']:
                        response['context'] = context
                        response['service_used'] = service_name
                        return response
                except Exception as e:
                    logger.warning(f"AI service {service_name} failed: {e}")
                    continue
            
            # If all services failed
            return {
                'success': False,
                'error': 'All configured AI services failed. Please check your API keys and network connection.',
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Error in AI chat: {e}")
            return {
                'success': False,
                'error': f'AI chat error: {str(e)}',
                'context': {}
            }
    
    def _get_available_ai_services(self) -> List[str]:
        """Get list of available AI services"""
        available = []
        
        if self.config_manager.is_ai_tool_enabled("anthropic"):
            api_key = self.config_manager.get_ai_tool_config("anthropic").get("api_key", "")
            if api_key.strip():
                available.append("anthropic")
        
        if self.config_manager.is_ai_tool_enabled("openai"):
            api_key = self.config_manager.get_ai_tool_config("openai").get("api_key", "")
            if api_key.strip():
                available.append("openai")
        
        if self.config_manager.is_ai_tool_enabled("local_models"):
            endpoint = self.config_manager.get_ai_tool_config("local_models").get("endpoint", "")
            if endpoint.strip():
                available.append("local_models")
        
        return available
    
    def _call_ai_service(self, service_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call specific AI service"""
        if service_name == "anthropic":
            return self._call_anthropic(context)
        elif service_name == "openai":
            return self._call_openai(context)
        elif service_name == "local_models":
            return self._call_local_model(context)
        else:
            raise ValueError(f"Unknown AI service: {service_name}")
    
    def _call_anthropic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            
            config = self.config_manager.get_ai_tool_config("anthropic")
            api_key = config.get("api_key", "")
            model = config.get("model", "claude-3-sonnet-20240229")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            # Build prompt with context
            prompt = self._build_anthropic_prompt(context)
            
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                'success': True,
                'response': response.content[0].text,
                'model': model,
                'service': 'Anthropic Claude'
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'Anthropic library not installed. Run: pip install anthropic'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Anthropic API error: {str(e)}'
            }
    
    def _call_openai(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI GPT API"""
        try:
            import openai
            
            config = self.config_manager.get_ai_tool_config("openai")
            api_key = config.get("api_key", "")
            model = config.get("model", "gpt-4")
            
            client = openai.OpenAI(api_key=api_key)
            
            # Build prompt with context
            prompt = self._build_openai_prompt(context)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant. Analyze the provided data and answer questions accurately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                'success': True,
                'response': response.choices[0].message.content,
                'model': model,
                'service': 'OpenAI GPT'
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'OpenAI library not installed. Run: pip install openai'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'OpenAI API error: {str(e)}'
            }
    
    def _call_local_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call local Ollama model"""
        try:
            import requests
            
            config = self.config_manager.get_ai_tool_config("local_models")
            endpoint = config.get("endpoint", "http://localhost:11434")
            model = config.get("default_model", "qwen2.5:3b")
            temperature = config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens", 512)
            
            # Build prompt with context
            prompt = self._build_local_prompt(context)
            
            response = requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result.get('response', 'No response received'),
                    'model': model,
                    'service': 'Local Ollama'
                }
            else:
                return {
                    'success': False,
                    'error': f'Ollama API error: {response.status_code} - {response.text}'
                }
                
        except ImportError:
            return {
                'success': False,
                'error': 'Requests library not installed. Run: pip install requests'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Local model error: {str(e)}'
            }
    
    def _build_anthropic_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Anthropic Claude"""
        return self._build_generic_prompt(context, "Claude")
    
    def _build_openai_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for OpenAI GPT"""
        return self._build_generic_prompt(context, "GPT")
    
    def _build_local_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for local model"""
        return self._build_generic_prompt(context, "Assistant")
    
    def _build_generic_prompt(self, context: Dict[str, Any], assistant_name: str) -> str:
        """Build generic prompt with context"""
        prompt_parts = []
        
        prompt_parts.append(f"You are {assistant_name}, a helpful data analyst assistant.")
        prompt_parts.append(f"User Question: {context['user_question']}")
        prompt_parts.append("")
        
        # Add context information
        if context['selected_items_count'] > 0:
            prompt_parts.append(f"Context: The user has selected {context['selected_items_count']} items for analysis:")
            prompt_parts.append("")
            
            # Add file information
            for file_ctx in context.get('files', []):
                prompt_parts.append(f"FILE: {file_ctx['name']}")
                prompt_parts.append(f"Type: {file_ctx['type']}")
                if file_ctx.get('summary'):
                    prompt_parts.append(f"Summary: {file_ctx['summary']}")
                if file_ctx.get('schema'):
                    prompt_parts.append(f"Schema: {file_ctx['schema']}")
                if file_ctx.get('content_preview'):
                    prompt_parts.append(f"Content Preview:\n{file_ctx['content_preview'][:1000]}")
                prompt_parts.append("")
            
            # Add dataset information
            for dataset_ctx in context.get('datasets', []):
                prompt_parts.append(f"DATASET: {dataset_ctx['name']}")
                prompt_parts.append(f"Type: {dataset_ctx['type']}")
                if dataset_ctx.get('description'):
                    prompt_parts.append(f"Description: {dataset_ctx['description']}")
                if dataset_ctx.get('schema_info'):
                    prompt_parts.append(f"Schema: {dataset_ctx['schema_info']}")
                if dataset_ctx.get('data_sample'):
                    sample = dataset_ctx['data_sample']
                    prompt_parts.append(f"Sample Data ({sample.get('note', 'sample')}):")
                    if sample.get('data'):
                        for i, row in enumerate(sample['data'][:5]):  # Limit to 5 rows
                            prompt_parts.append(f"  Row {i+1}: {row}")
                prompt_parts.append("")
        
        prompt_parts.append("Please analyze the provided data and answer the user's question accurately and helpfully.")
        prompt_parts.append("If you need more information or if something is unclear, please ask for clarification.")
        
        return '\n'.join(prompt_parts)
