"""
AI Chat Service
Handles AI conversations with context from selected files and datasets
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import datetime

logger = logging.getLogger(__name__)

def log_prompt_to_file(prompt: str, model: str, service: str, user_question: str):
    """Log the AI prompt to a file for debugging"""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_prompt_{timestamp}_{service}_{model.replace(':', '_')}.txt"
        log_file = logs_dir / filename
        
        # Write prompt to file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"AI CHAT PROMPT LOG\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Service: {service}\n")
            f.write(f"Model: {model}\n")
            f.write(f"User Question: {user_question}\n")
            f.write("=" * 80 + "\n\n")
            f.write("FULL PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt)
            f.write("\n" + "-" * 40 + "\n")
            f.write(f"Prompt Length: {len(prompt)} characters\n")
            f.write(f"Estimated Tokens: {len(prompt) // 4}\n")
        
        logger.info(f"Prompt logged to: {log_file}")
        
    except Exception as e:
        logger.error(f"Failed to log prompt to file: {e}")

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
            logger.info(f"=== AI CHAT REQUEST START ===")
            logger.info(f"User question: {user_question}")
            logger.info(f"Selected items count: {len(selected_items)}")
            
            # Log selected items details
            for i, item in enumerate(selected_items):
                logger.info(f"Selected item {i+1}: {item.get('name', 'Unknown')} ({item.get('type', 'Unknown')})")
            
            # Build context from selected items
            context = self.context_builder.build_context(selected_items, user_question)
            logger.info(f"Built context with {context['selected_items_count']} items")
            logger.info(f"Context token estimate: {context.get('total_tokens_estimate', 0)}")
            
            # Check which AI services are available
            available_services = self._get_available_ai_services()
            logger.info(f"Available AI services: {available_services}")
            
            if not available_services:
                return {
                    'success': False,
                    'error': 'No AI services configured or available. Please check Settings → AI Tools and ensure at least one service is properly configured.',
                    'context': context
                }
            
            # Collect all errors for detailed reporting
            service_errors = {}
            
            # Try each available service
            for service_name in available_services:
                try:
                    logger.info(f"Trying AI service: {service_name}")
                    response = self._call_ai_service(service_name, context)
                    
                    if response['success']:
                        logger.info(f"Successfully got response from {service_name}")
                        logger.info(f"Response length: {len(response.get('response', ''))} characters")
                        logger.info(f"Model used: {response.get('model', 'Unknown')}")
                        logger.info(f"=== AI CHAT REQUEST SUCCESS ===")
                        response['context'] = context
                        response['service_used'] = service_name
                        return response
                    else:
                        service_errors[service_name] = response.get('error', 'Unknown error')
                        logger.warning(f"AI service {service_name} failed: {response.get('error')}")
                        
                except Exception as e:
                    error_msg = str(e)
                    service_errors[service_name] = error_msg
                    logger.error(f"AI service {service_name} exception: {e}")
                    continue
            
            # If all services failed, provide detailed error information
            error_details = []
            for service, error in service_errors.items():
                error_details.append(f"• {service}: {error}")
            
            detailed_error = f"All configured AI services failed:\n\n" + "\n".join(error_details)
            detailed_error += f"\n\nTroubleshooting:\n"
            detailed_error += f"1. Check your internet connection\n"
            detailed_error += f"2. Verify API keys in Settings → AI Tools\n"
            detailed_error += f"3. For local models, ensure Ollama is running: ollama serve\n"
            detailed_error += f"4. Check the application logs for more details"
            
            logger.error(f"=== AI CHAT REQUEST FAILED ===")
            logger.error(f"All services failed. Errors: {service_errors}")
            
            return {
                'success': False,
                'error': detailed_error,
                'context': context,
                'service_errors': service_errors
            }
            
        except Exception as e:
            logger.error(f"=== AI CHAT REQUEST ERROR ===")
            logger.error(f"System error in AI chat: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': f'AI chat system error: {str(e)}',
                'context': {}
            }
    
    def _get_available_ai_services(self) -> List[str]:
        """Get list of available AI services"""
        available = []
        
        # Check Anthropic Claude
        try:
            if self.config_manager.is_ai_tool_enabled("anthropic"):
                api_key = self.config_manager.get_ai_tool_config("anthropic").get("api_key", "")
                if api_key.strip():
                    available.append("anthropic")
                    logger.info("Anthropic Claude service available")
        except Exception as e:
            logger.warning(f"Error checking Anthropic availability: {e}")
        
        # Check OpenAI GPT
        try:
            if self.config_manager.is_ai_tool_enabled("openai"):
                api_key = self.config_manager.get_ai_tool_config("openai").get("api_key", "")
                if api_key.strip():
                    available.append("openai")
                    logger.info("OpenAI GPT service available")
        except Exception as e:
            logger.warning(f"Error checking OpenAI availability: {e}")
        
        # Check Local Models (Ollama) with automatic setup
        try:
            if self.config_manager.is_ai_tool_enabled("local_models"):
                config = self.config_manager.get_ai_tool_config("local_models")
                endpoint = config.get("endpoint", "http://localhost:11434")
                model = config.get("default_model", "qwen2.5:3b")
                
                if endpoint.strip() and model.strip():
                    # Use Ollama manager to check availability
                    from utils.ollama_manager import get_ollama_manager
                    
                    ollama_manager = get_ollama_manager(self.config_manager)
                    status = ollama_manager.get_status()
                    
                    if status['ollama_available']:
                        # Ollama is installed, mark as available
                        # The actual model setup will happen when needed
                        available.append("local_models")
                        logger.info(f"Local Ollama available (will auto-setup {model} when needed)")
                    else:
                        logger.warning(f"Ollama not available: {status.get('error', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"Error checking local models availability: {e}")
        
        logger.info(f"Available AI services: {available}")
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
            
            logger.info(f"=== ANTHROPIC REQUEST ===")
            logger.info(f"Model: {model}")
            logger.info(f"API key configured: {'Yes' if api_key else 'No'}")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            # Build prompt with context
            prompt = self._build_anthropic_prompt(context)
            logger.info(f"Generated prompt length: {len(prompt)} characters")
            logger.debug(f"Full prompt content:\n{prompt}")
            
            # Log prompt to file
            log_prompt_to_file(prompt, model, "Anthropic", context['user_question'])
            
            logger.info(f"Sending request to Anthropic Claude API")
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            ai_response = response.content[0].text
            logger.info(f"Anthropic response length: {len(ai_response)} characters")
            logger.debug(f"Anthropic response content: {ai_response}")
            
            return {
                'success': True,
                'response': ai_response,
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
            
            logger.info(f"=== OPENAI REQUEST ===")
            logger.info(f"Model: {model}")
            logger.info(f"API key configured: {'Yes' if api_key else 'No'}")
            
            client = openai.OpenAI(api_key=api_key)
            
            # Build prompt with context
            prompt = self._build_openai_prompt(context)
            logger.info(f"Generated prompt length: {len(prompt)} characters")
            logger.debug(f"Full prompt content:\n{prompt}")
            
            # Log prompt to file
            log_prompt_to_file(prompt, model, "OpenAI", context['user_question'])
            
            logger.info(f"Sending request to OpenAI GPT API")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant. Analyze the provided data and answer questions accurately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"OpenAI response length: {len(ai_response)} characters")
            logger.debug(f"OpenAI response content: {ai_response}")
            
            return {
                'success': True,
                'response': ai_response,
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
        """Call local Ollama model with automatic setup"""
        try:
            import requests
            from utils.ollama_manager import get_ollama_manager
            
            config = self.config_manager.get_ai_tool_config("local_models")
            endpoint = config.get("endpoint", "http://localhost:11434")
            model = config.get("default_model", "qwen2.5:3b")
            temperature = config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens", 512)
            
            logger.info(f"=== LOCAL MODEL REQUEST ===")
            logger.info(f"Model: {model}")
            logger.info(f"Endpoint: {endpoint}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Max tokens: {max_tokens}")
            
            # Get Ollama manager and ensure model is ready
            ollama_manager = get_ollama_manager(self.config_manager)
            
            # Ensure model is pulled and ready
            setup_result = ollama_manager.ensure_model_ready(model, endpoint)
            if not setup_result['success']:
                return {
                    'success': False,
                    'error': f'Failed to prepare model {model}: {setup_result["error"]}',
                    'suggestion': setup_result.get('suggestion', 'Check Ollama installation')
                }
            
            logger.info(f"Model {model} is ready, making request...")
            
            # Build prompt with context
            prompt = self._build_local_prompt(context)
            logger.info(f"Generated prompt length: {len(prompt)} characters")
            logger.debug(f"Full prompt content:\n{prompt}")
            
            # Log prompt to file
            log_prompt_to_file(prompt, model, "Local_Ollama", context['user_question'])
            
            # Make the generation request
            generation_payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            logger.info(f"Sending request to {endpoint}/api/generate")
            logger.debug(f"Request payload: {generation_payload}")
            
            response = requests.post(
                f"{endpoint}/api/generate",
                json=generation_payload,
                timeout=60  # Increased timeout for generation
            )
            
            logger.info(f"Ollama response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Ollama response text: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'No response received')
                
                logger.info(f"Raw response from Ollama: {result}")
                logger.info(f"AI response length: {len(ai_response)} characters")
                logger.debug(f"AI response content: {ai_response}")
                
                if not ai_response or ai_response.strip() == '':
                    logger.warning(f"Model {model} returned empty response")
                    return {
                        'success': False,
                        'error': f'Model {model} returned empty response. The model may not be properly loaded.'
                    }
                
                logger.info(f"Successfully received response from {model}")
                return {
                    'success': True,
                    'response': ai_response,
                    'model': model,
                    'service': 'Local Ollama'
                }
            else:
                error_text = response.text
                logger.error(f"Ollama API error: {response.status_code} - {error_text}")
                return {
                    'success': False,
                    'error': f'Ollama API error: {response.status_code} - {error_text}'
                }
                
        except ImportError:
            return {
                'success': False,
                'error': 'Required libraries not installed. Run: pip install requests'
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': f'Request to {model} timed out. The model may be loading or the query is too complex.'
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Network error connecting to Ollama: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error in local model call: {e}")
            return {
                'success': False,
                'error': f'Unexpected local model error: {str(e)}'
            }
    
    def _build_anthropic_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for Anthropic Claude"""
        return self._build_generic_prompt(context, "Claude")
    
    def _build_openai_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for OpenAI GPT"""
        return self._build_generic_prompt(context, "GPT")
    
    def _build_local_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for local model with enhanced context formatting"""
        return self._build_enhanced_local_prompt(context)
    
    def _build_enhanced_local_prompt(self, context: Dict[str, Any]) -> str:
        """Build enhanced prompt specifically optimized for local models like Gemma"""
        prompt_parts = []
        
        # Clear system instruction for local models
        prompt_parts.append("You are a helpful data analyst assistant. Analyze the provided data context and answer the user's question accurately.")
        prompt_parts.append("")
        
        # Emphasize the user's question prominently
        prompt_parts.append("=== USER QUESTION ===")
        prompt_parts.append(context['user_question'])
        prompt_parts.append("")
        
        # Add context information with clear structure
        if context['selected_items_count'] > 0:
            prompt_parts.append("=== DATA CONTEXT ===")
            prompt_parts.append(f"The user has selected {context['selected_items_count']} data items for analysis:")
            prompt_parts.append("")
            
            # Process files with enhanced formatting
            for i, file_ctx in enumerate(context.get('files', []), 1):
                prompt_parts.append(f"--- FILE {i}: {file_ctx['name']} ---")
                prompt_parts.append(f"File Type: {file_ctx['type']}")
                prompt_parts.append(f"Location: {file_ctx.get('path', 'Unknown')}")
                
                if file_ctx.get('summary'):
                    prompt_parts.append(f"Summary: {file_ctx['summary']}")
                
                if file_ctx.get('schema'):
                    schema = file_ctx['schema']
                    if isinstance(schema, dict):
                        if 'columns' in schema:
                            prompt_parts.append(f"Columns: {', '.join(schema['columns'])}")
                        if 'sample_data' in schema and schema['sample_data']:
                            prompt_parts.append("Sample Data:")
                            for j, row in enumerate(schema['sample_data'][:3]):
                                prompt_parts.append(f"  Row {j+1}: {row}")
                    else:
                        prompt_parts.append(f"Schema: {schema}")
                
                if file_ctx.get('content_preview'):
                    content = file_ctx['content_preview'][:800]  # Limit for local models
                    prompt_parts.append(f"Content Preview:")
                    prompt_parts.append(content)
                
                prompt_parts.append("")
            
            # Process datasets with enhanced formatting
            for i, dataset_ctx in enumerate(context.get('datasets', []), 1):
                prompt_parts.append(f"--- DATASET {i}: {dataset_ctx['name']} ---")
                prompt_parts.append(f"Dataset Type: {dataset_ctx['type']}")
                
                if dataset_ctx.get('description'):
                    prompt_parts.append(f"Description: {dataset_ctx['description']}")
                
                if dataset_ctx.get('tags'):
                    tags = dataset_ctx['tags']
                    if isinstance(tags, list):
                        prompt_parts.append(f"Tags: {', '.join(tags)}")
                    else:
                        prompt_parts.append(f"Tags: {tags}")
                
                if dataset_ctx.get('schema_info'):
                    prompt_parts.append(f"Schema Information: {dataset_ctx['schema_info']}")
                
                if dataset_ctx.get('data_sample'):
                    sample = dataset_ctx['data_sample']
                    prompt_parts.append(f"Sample Data ({sample.get('note', 'sample')}):")
                    if sample.get('columns'):
                        prompt_parts.append(f"Columns: {', '.join(sample['columns'])}")
                    if sample.get('data'):
                        prompt_parts.append("Sample Rows:")
                        for j, row in enumerate(sample['data'][:3]):  # Limit to 3 rows for local models
                            if isinstance(row, dict):
                                row_str = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:5]])
                                prompt_parts.append(f"  Row {j+1}: {row_str}")
                            else:
                                prompt_parts.append(f"  Row {j+1}: {row}")
                
                prompt_parts.append("")
        else:
            prompt_parts.append("=== NO SPECIFIC DATA CONTEXT ===")
            prompt_parts.append("The user is asking a general question without selecting specific data items.")
            prompt_parts.append("")
        
        # Clear instruction for response
        prompt_parts.append("=== INSTRUCTIONS ===")
        prompt_parts.append("Based on the data context provided above, please:")
        prompt_parts.append("1. Analyze the relevant data items")
        prompt_parts.append("2. Answer the user's question accurately and helpfully")
        prompt_parts.append("3. Reference specific data points when relevant")
        prompt_parts.append("4. If you need clarification, ask specific questions")
        prompt_parts.append("")
        prompt_parts.append("Response:")
        
        return '\n'.join(prompt_parts)
    
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
