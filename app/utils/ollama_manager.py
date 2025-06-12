"""
Ollama Model Manager
Handles automatic pulling, serving, and lifecycle management of Ollama models
"""

import logging
import subprocess
import time
import requests
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import psutil
import signal
import os

logger = logging.getLogger(__name__)

class OllamaManager:
    """Manages Ollama service and model lifecycle"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.ollama_process = None
        self.is_serving = False
        self.served_models = set()
        self.shutdown_requested = False
        
    def ensure_ollama_available(self) -> Dict[str, Any]:
        """Ensure Ollama is installed and available"""
        try:
            # Check if ollama command is available
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Ollama found: {version}")
                return {'success': True, 'version': version}
            else:
                return {
                    'success': False, 
                    'error': 'Ollama command failed',
                    'suggestion': 'Please install Ollama from https://ollama.ai'
                }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Ollama command timed out',
                'suggestion': 'Ollama may be unresponsive. Try restarting it.'
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'Ollama not found in PATH',
                'suggestion': 'Please install Ollama from https://ollama.ai and ensure it\'s in your PATH'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error checking Ollama: {str(e)}',
                'suggestion': 'Please check your Ollama installation'
            }
    
    def is_ollama_serving(self, endpoint: str = "http://localhost:11434") -> bool:
        """Check if Ollama is currently serving"""
        try:
            response = requests.get(f"{endpoint}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama_service(self, endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
        """Start Ollama service if not already running"""
        try:
            # First check if already running
            if self.is_ollama_serving(endpoint):
                logger.info("Ollama is already serving")
                self.is_serving = True
                return {'success': True, 'message': 'Ollama already running'}
            
            # Check if Ollama is available
            availability = self.ensure_ollama_available()
            if not availability['success']:
                return availability
            
            logger.info("Starting Ollama service...")
            
            # Start Ollama serve in background
            if os.name == 'nt':  # Windows
                # Use CREATE_NEW_PROCESS_GROUP to allow clean shutdown
                self.ollama_process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Unix-like
                self.ollama_process = subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            # Wait for service to start (up to 30 seconds)
            for i in range(30):
                time.sleep(1)
                if self.is_ollama_serving(endpoint):
                    logger.info(f"Ollama service started successfully (took {i+1} seconds)")
                    self.is_serving = True
                    return {'success': True, 'message': f'Ollama started in {i+1} seconds'}
                
                # Check if process died
                if self.ollama_process.poll() is not None:
                    stdout, stderr = self.ollama_process.communicate()
                    return {
                        'success': False,
                        'error': f'Ollama process died: {stderr.decode()}',
                        'suggestion': 'Check Ollama installation and try again'
                    }
            
            # Timeout
            return {
                'success': False,
                'error': 'Ollama service failed to start within 30 seconds',
                'suggestion': 'Try starting Ollama manually: ollama serve'
            }
            
        except Exception as e:
            logger.error(f"Error starting Ollama service: {e}")
            return {
                'success': False,
                'error': f'Failed to start Ollama: {str(e)}',
                'suggestion': 'Check Ollama installation and permissions'
            }
    
    def get_available_models(self, endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
        """Get list of available models"""
        try:
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return {'success': True, 'models': models}
            else:
                return {
                    'success': False,
                    'error': f'API returned status {response.status_code}',
                    'models': []
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error getting models: {str(e)}',
                'models': []
            }
    
    def pull_model(self, model_name: str, endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
        """Pull a model if not already available"""
        try:
            # First check if model is already available
            models_result = self.get_available_models(endpoint)
            if models_result['success'] and model_name in models_result['models']:
                logger.info(f"Model {model_name} already available")
                return {'success': True, 'message': f'Model {model_name} already available'}
            
            logger.info(f"Pulling model {model_name}...")
            
            # Use subprocess to pull model with real-time output
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor the pull process
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    logger.info(f"Ollama pull: {output.strip()}")
            
            return_code = process.poll()
            
            if return_code == 0:
                logger.info(f"Successfully pulled model {model_name}")
                return {
                    'success': True,
                    'message': f'Successfully pulled {model_name}',
                    'output': '\n'.join(output_lines)
                }
            else:
                error_output = '\n'.join(output_lines)
                logger.error(f"Failed to pull model {model_name}: {error_output}")
                return {
                    'success': False,
                    'error': f'Failed to pull {model_name}',
                    'output': error_output,
                    'suggestion': f'Check if model name "{model_name}" is correct'
                }
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return {
                'success': False,
                'error': f'Error pulling model: {str(e)}',
                'suggestion': 'Check your internet connection and Ollama installation'
            }
    
    def ensure_model_ready(self, model_name: str, endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
        """Ensure model is pulled and ready for use"""
        try:
            # Step 1: Ensure Ollama service is running
            if not self.is_ollama_serving(endpoint):
                start_result = self.start_ollama_service(endpoint)
                if not start_result['success']:
                    return start_result
            
            # Step 2: Check if model is available
            models_result = self.get_available_models(endpoint)
            if not models_result['success']:
                return models_result
            
            # Step 3: Pull model if not available
            if model_name not in models_result['models']:
                pull_result = self.pull_model(model_name, endpoint)
                if not pull_result['success']:
                    return pull_result
            
            # Step 4: Test model by making a simple request
            test_result = self.test_model(model_name, endpoint)
            if test_result['success']:
                self.served_models.add(model_name)
                return {
                    'success': True,
                    'message': f'Model {model_name} is ready for use',
                    'model': model_name
                }
            else:
                return test_result
                
        except Exception as e:
            logger.error(f"Error ensuring model ready: {e}")
            return {
                'success': False,
                'error': f'Error preparing model: {str(e)}',
                'suggestion': 'Check Ollama installation and model name'
            }
    
    def test_model(self, model_name: str, endpoint: str = "http://localhost:11434") -> Dict[str, Any]:
        """Test if model is working by making a simple request"""
        try:
            logger.info(f"Testing model {model_name}...")
            
            test_payload = {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_predict": 5  # Very short response for testing
                }
            }
            
            response = requests.post(
                f"{endpoint}/api/generate",
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('response'):
                    logger.info(f"Model {model_name} test successful")
                    return {
                        'success': True,
                        'message': f'Model {model_name} is working correctly'
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Model {model_name} returned empty response',
                        'suggestion': 'Model may not be properly loaded'
                    }
            else:
                return {
                    'success': False,
                    'error': f'Model test failed with status {response.status_code}',
                    'suggestion': f'Check if model {model_name} is properly installed'
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': f'Model {model_name} test timed out',
                'suggestion': 'Model may be loading or system is slow'
            }
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {e}")
            return {
                'success': False,
                'error': f'Error testing model: {str(e)}',
                'suggestion': 'Check Ollama service and model installation'
            }
    
    def shutdown_ollama_service(self):
        """Shutdown Ollama service if we started it"""
        if not self.ollama_process:
            logger.info("No Ollama process to shutdown")
            return
        
        try:
            self.shutdown_requested = True
            logger.info("Shutting down Ollama service...")
            
            if os.name == 'nt':  # Windows
                # Send CTRL_BREAK_EVENT to the process group
                self.ollama_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:  # Unix-like
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.ollama_process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                self.ollama_process.wait(timeout=10)
                logger.info("Ollama service shut down gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Ollama service didn't shut down gracefully, forcing termination")
                if os.name == 'nt':
                    self.ollama_process.terminate()
                else:
                    os.killpg(os.getpgid(self.ollama_process.pid), signal.SIGKILL)
                self.ollama_process.wait()
            
            self.ollama_process = None
            self.is_serving = False
            self.served_models.clear()
            
        except Exception as e:
            logger.error(f"Error shutting down Ollama service: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of Ollama manager"""
        config = self.config_manager.get_ai_tool_config("local_models")
        endpoint = config.get("endpoint", "http://localhost:11434")
        
        status = {
            'ollama_available': False,
            'service_running': False,
            'models_available': [],
            'served_models': list(self.served_models),
            'process_managed': self.ollama_process is not None,
            'endpoint': endpoint
        }
        
        # Check Ollama availability
        availability = self.ensure_ollama_available()
        status['ollama_available'] = availability['success']
        if not availability['success']:
            status['error'] = availability['error']
            return status
        
        # Check service status
        status['service_running'] = self.is_ollama_serving(endpoint)
        
        # Get available models
        if status['service_running']:
            models_result = self.get_available_models(endpoint)
            if models_result['success']:
                status['models_available'] = models_result['models']
        
        return status

# Global instance
_ollama_manager = None

def get_ollama_manager(config_manager) -> OllamaManager:
    """Get global Ollama manager instance"""
    global _ollama_manager
    if _ollama_manager is None:
        _ollama_manager = OllamaManager(config_manager)
    return _ollama_manager

def shutdown_ollama():
    """Shutdown global Ollama manager"""
    global _ollama_manager
    if _ollama_manager:
        _ollama_manager.shutdown_ollama_service()
        _ollama_manager = None
