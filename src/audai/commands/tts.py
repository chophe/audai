"""Text-to-Speech (TTS) command implementation."""

import os
from pathlib import Path
from typing import Optional

import typer
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..config import config
from ..audio_utils import play_audio, generate_audio_filename

console = Console()
tts_command = typer.Typer(help="🗣️ Text-to-Speech commands")

class TTSProvider:
    """Base class for TTS providers."""
    
    def __init__(self, model_config):
        self.config = model_config
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """Synthesize text to speech.
        
        Args:
            text: Text to synthesize.
            output_path: Output file path.
            
        Returns:
            True if successful, False otherwise.
        """
        raise NotImplementedError

class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider."""
    
    def __init__(self, model_config):
        super().__init__(model_config)
        if not OpenAI:
            raise ImportError("openai package is required for OpenAI TTS")
        
        self.client = OpenAI(
            api_key=model_config.api_key,
            base_url=model_config.base_url
        )
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """Synthesize text using OpenAI TTS."""
        try:
            with console.status("[bold green]Generating speech..."):
                response = self.client.audio.speech.create(
                    model=self.config.model,
                    voice=self.config.voice or "alloy",
                    input=text
                )
            
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            console.print(f"[red]OpenAI TTS error: {e}[/red]")
            return False

class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS provider."""
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """Synthesize text using ElevenLabs TTS."""
        try:
            url = f"{self.config.base_url}/text-to-speech/{os.getenv('ELEVENLABS_VOICE_ID')}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.config.api_key
            }
            
            data = {
                "text": text,
                "model_id": self.config.model,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            with console.status("[bold green]Generating speech..."):
                response = requests.post(url, json=data, headers=headers)
                response.raise_for_status()
            
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            console.print(f"[red]ElevenLabs TTS error: {e}[/red]")
            return False

class LocalTTSProvider(TTSProvider):
    """Local/Custom TTS provider."""
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """Synthesize text using local TTS service."""
        try:
            url = f"{self.config.base_url}/tts"
            
            headers = {"Content-Type": "application/json"}
            data = {
                "text": text,
                "model": self.config.model,
                "voice": getattr(self.config, 'voice', 'default')
            }
            
            # Add API key if available
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            with console.status("[bold green]Generating speech..."):
                response = requests.post(url, json=data, headers=headers, timeout=30)
                response.raise_for_status()
            
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            console.print(f"[red]Local TTS error: {e}[/red]")
            return False

def get_tts_provider(model_config) -> TTSProvider:
    """Get TTS provider based on model configuration.
    
    Args:
        model_config: Model configuration object.
        
    Returns:
        TTS provider instance.
    """
    provider_map = {
        'openai': OpenAITTSProvider,
        'azure': OpenAITTSProvider,  # Azure uses OpenAI-compatible API
        'elevenlabs': ElevenLabsTTSProvider,
        'local': LocalTTSProvider
    }
    
    provider_class = provider_map.get(model_config.provider)
    if not provider_class:
        raise ValueError(f"Unsupported TTS provider: {model_config.provider}")
    
    return provider_class(model_config)

@tts_command.command()
def synthesize(
    text: str = typer.Argument(..., help="Text to convert to speech"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="TTS model to use"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice to use for synthesis"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    play: bool = typer.Option(False, "--play", "-p", help="Play the generated audio"),
    format: str = typer.Option("mp3", "--format", "-f", help="Audio format (mp3, wav, etc.)"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output")
):
    """🗣️ Convert text to speech using AI models.
    
    Examples:
        audai tts "Hello, world!"
        audai tts "Hello, world!" --play
        audai tts "Hello, world!" --model openai_tts_hd --voice nova
        audai tts "Hello, world!" --output my_speech.mp3
    """
    # Get model configuration
    model_config = config.get_tts_model(model)
    if not model_config:
        available_models = config.list_models('tts')
        console.print(f"[red]TTS model '{model or config.default_tts_model}' not found.[/red]")
        console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
        raise typer.Exit(1)
    
    # Override voice if specified
    if voice:
        model_config.voice = voice
    
    # Generate output filename if not specified
    if not output:
        output = generate_audio_filename(config.output_dir, "tts", format)
    else:
        output = str(Path(output).resolve())
    
    # Ensure output directory exists
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        console.print(f"[dim]Using model: {model_config.name} ({model_config.provider})[/dim]")
        console.print(f"[dim]Output file: {output}[/dim]")
        if hasattr(model_config, 'voice') and model_config.voice:
            console.print(f"[dim]Voice: {model_config.voice}[/dim]")
    
    # Display text being synthesized
    text_display = Text(text)
    if len(text) > 100:
        text_display = Text(text[:97] + "...")
    
    console.print(Panel(
        text_display,
        title="Text to Synthesize",
        border_style="blue"
    ))
    
    try:
        # Get TTS provider and synthesize
        provider = get_tts_provider(model_config)
        
        success = provider.synthesize(text, output)
        
        if success:
            # Check if file was created and has content
            if os.path.exists(output) and os.path.getsize(output) > 0:
                file_size = os.path.getsize(output) / 1024  # KB
                console.print(f"[green]✓ Speech generated successfully![/green]")
                console.print(f"[dim]File: {output} ({file_size:.1f} KB)[/dim]")
                
                # Play audio if requested
                if play:
                    console.print("[blue]🔊 Playing audio...[/blue]")
                    if not play_audio(output):
                        console.print("[yellow]Could not play audio automatically. File saved successfully.[/yellow]")
            else:
                console.print("[red]✗ Audio file was not created or is empty[/red]")
                raise typer.Exit(1)
        else:
            console.print("[red]✗ Failed to generate speech[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during synthesis: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)

@tts_command.command("voices")
def list_voices(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="TTS model to show voices for")
):
    """🎭 List available voices for TTS models."""
    model_config = config.get_tts_model(model)
    if not model_config:
        available_models = config.list_models('tts')
        console.print(f"[red]TTS model '{model or config.default_tts_model}' not found.[/red]")
        console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
        raise typer.Exit(1)
    
    # Common voices for different providers
    voice_info = {
        'openai': {
            'alloy': 'Neutral, balanced voice',
            'echo': 'Clear, expressive voice',
            'fable': 'Warm, engaging voice',
            'onyx': 'Deep, authoritative voice',
            'nova': 'Bright, energetic voice',
            'shimmer': 'Soft, gentle voice'
        },
        'elevenlabs': {
            'default': 'Default ElevenLabs voice (configure ELEVENLABS_VOICE_ID)'
        },
        'local': {
            'default': 'Default local voice (depends on your TTS service)'
        }
    }
    
    provider_voices = voice_info.get(model_config.provider, {'default': 'Default voice'})
    
    console.print(f"[bold]Available voices for {model_config.name} ({model_config.provider}):[/bold]\n")
    
    for voice_id, description in provider_voices.items():
        current_marker = " [dim](current)[/dim]" if voice_id == getattr(model_config, 'voice', 'alloy') else ""
        console.print(f"• [cyan]{voice_id}[/cyan]{current_marker}: {description}")
    
    console.print(f"\n[dim]Use --voice option to specify a voice: audai tts 'Hello' --voice nova[/dim]")

@tts_command.command("test")
def test_tts(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="TTS model to test"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice to test"),
    play: bool = typer.Option(True, "--play/--no-play", help="Play the test audio")
):
    """🧪 Test TTS functionality with a sample text."""
    test_text = "Hello! This is a test of the text-to-speech functionality. How does it sound?"
    
    console.print(Panel(
        "[bold blue]Testing TTS functionality[/bold blue]\n\n"
        f"Sample text: {test_text}",
        title="TTS Test",
        border_style="blue"
    ))
    
    # Use the synthesize command with test parameters
    ctx = typer.Context(synthesize)
    ctx.invoke(
        synthesize,
        text=test_text,
        model=model,
        voice=voice,
        play=play,
        verbose=True
    )