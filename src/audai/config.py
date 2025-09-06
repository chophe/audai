"""Configuration management for AudAI CLI."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: str
    model: str
    voice: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Set base_url and api_key based on provider if not explicitly set."""
        if not self.base_url:
            self.base_url = self._get_provider_base_url()
        if not self.api_key:
            self.api_key = self._get_provider_api_key()
    
    def _get_provider_base_url(self) -> Optional[str]:
        """Get base URL for the provider."""
        provider_urls = {
            'openai': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'azure': os.getenv('AZURE_OPENAI_BASE_URL'),
            'elevenlabs': os.getenv('ELEVENLABS_BASE_URL', 'https://api.elevenlabs.io/v1'),
            'google': os.getenv('GOOGLE_BASE_URL', 'https://speech.googleapis.com/v1'),
            'deepgram': os.getenv('DEEPGRAM_BASE_URL', 'https://api.deepgram.com/v1'),
            'huggingface': os.getenv('HUGGINGFACE_BASE_URL', 'https://api-inference.huggingface.co'),
            'replicate': os.getenv('REPLICATE_BASE_URL', 'https://api.replicate.com/v1'),
            'local': os.getenv(f'LOCAL_{self.provider.upper()}_BASE_URL', 'http://localhost:8000')
        }
        return provider_urls.get(self.provider)
    
    def _get_provider_api_key(self) -> Optional[str]:
        """Get API key for the provider."""
        provider_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'azure': os.getenv('AZURE_OPENAI_API_KEY'),
            'elevenlabs': os.getenv('ELEVENLABS_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY'),
            'deepgram': os.getenv('DEEPGRAM_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_API_KEY'),
            'replicate': os.getenv('REPLICATE_API_TOKEN'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY')
        }
        return provider_keys.get(self.provider)

class Config:
    """Main configuration class for AudAI."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
        """
        self.env_file = env_file or '.env'
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment variables."""
        # Load .env file if it exists
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
        else:
            console.print(Panel(
                f"[yellow]Warning: {self.env_file} not found. Using environment variables only.[/yellow]\n"
                f"[dim]Copy .env.template to {self.env_file} and configure your settings.[/dim]",
                title="Configuration",
                border_style="yellow"
            ))
        
        # Basic settings
        self.output_dir = Path(os.getenv('OUTPUT_DIR', './output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Default models
        self.default_tts_model = os.getenv('DEFAULT_TTS_MODEL', 'openai_tts')
        self.default_whisper_model = os.getenv('DEFAULT_WHISPER_MODEL', 'openai_whisper')
        self.default_talk_model = os.getenv('DEFAULT_TALK_MODEL', 'openai_realtime')
        
        # Audio settings
        self.audio_format = os.getenv('AUDIO_FORMAT', 'mp3')
        self.audio_sample_rate = int(os.getenv('AUDIO_SAMPLE_RATE', '44100'))
        self.audio_channels = int(os.getenv('AUDIO_CHANNELS', '1'))
        self.record_stop_key = os.getenv('RECORD_STOP_KEY', 'space')
        self.record_timeout = int(os.getenv('RECORD_TIMEOUT', '30'))
        
        # Load model configurations
        self.tts_models = self._load_models('MODELS_TTS')
        self.whisper_models = self._load_models('MODELS_WHISPER')
        self.talk_models = self._load_models('MODELS_TALK')
    
    def _load_models(self, env_var: str) -> Dict[str, ModelConfig]:
        """Load model configurations from environment variable.
        
        Args:
            env_var: Environment variable name containing JSON model definitions.
            
        Returns:
            Dictionary mapping model names to ModelConfig objects.
        """
        models_json = os.getenv(env_var, '[]')
        try:
            models_data = json.loads(models_json)
            models = {}
            for model_data in models_data:
                model = ModelConfig(**model_data)
                models[model.name] = model
            return models
        except (json.JSONDecodeError, TypeError) as e:
            console.print(f"[red]Error loading {env_var}: {e}[/red]")
            return {}
    
    def get_tts_model(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """Get TTS model configuration.
        
        Args:
            model_name: Name of the model. If None, uses default.
            
        Returns:
            ModelConfig object or None if not found.
        """
        name = model_name or self.default_tts_model
        return self.tts_models.get(name)
    
    def get_whisper_model(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """Get Whisper model configuration.
        
        Args:
            model_name: Name of the model. If None, uses default.
            
        Returns:
            ModelConfig object or None if not found.
        """
        name = model_name or self.default_whisper_model
        return self.whisper_models.get(name)
    
    def get_talk_model(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """Get Talk model configuration.
        
        Args:
            model_name: Name of the model. If None, uses default.
            
        Returns:
            ModelConfig object or None if not found.
        """
        name = model_name or self.default_talk_model
        return self.talk_models.get(name)
    
    def list_models(self, model_type: str) -> List[str]:
        """List available models of a specific type.
        
        Args:
            model_type: Type of models ('tts', 'whisper', 'talk').
            
        Returns:
            List of model names.
        """
        model_dict = {
            'tts': self.tts_models,
            'whisper': self.whisper_models,
            'talk': self.talk_models
        }
        return list(model_dict.get(model_type, {}).keys())
    
    def validate_config(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        issues = []
        
        # Check if output directory is writable
        if not os.access(self.output_dir, os.W_OK):
            issues.append(f"Output directory {self.output_dir} is not writable")
        
        # Check if default models exist
        if not self.get_tts_model():
            issues.append(f"Default TTS model '{self.default_tts_model}' not found")
        
        if not self.get_whisper_model():
            issues.append(f"Default Whisper model '{self.default_whisper_model}' not found")
        
        if not self.get_talk_model():
            issues.append(f"Default Talk model '{self.default_talk_model}' not found")
        
        if issues:
            console.print(Panel(
                "\n".join([f"[red]• {issue}[/red]" for issue in issues]),
                title="Configuration Issues",
                border_style="red"
            ))
            return False
        
        return True
    
    def show_config(self):
        """Display current configuration."""
        config_info = f"""
[bold]Output Directory:[/bold] {self.output_dir}
[bold]Audio Format:[/bold] {self.audio_format}
[bold]Sample Rate:[/bold] {self.audio_sample_rate} Hz
[bold]Channels:[/bold] {self.audio_channels}

[bold]Default Models:[/bold]
• TTS: {self.default_tts_model}
• Whisper: {self.default_whisper_model}
• Talk: {self.default_talk_model}

[bold]Available Models:[/bold]
• TTS: {', '.join(self.list_models('tts'))}
• Whisper: {', '.join(self.list_models('whisper'))}
• Talk: {', '.join(self.list_models('talk'))}
        """
        
        console.print(Panel(
            config_info,
            title="AudAI Configuration",
            border_style="blue"
        ))

# Global configuration instance
config = Config()