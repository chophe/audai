import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import openai
import requests
from datetime import datetime

from ..config import Config
from ..audio_utils import record_audio_with_key, validate_audio_file, get_audio_duration, AudioProgressCallback

console = Console()
app = typer.Typer(help="Audio transcription using Whisper models")

class WhisperProvider:
    """Base class for Whisper transcription providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'Unknown')
        self.provider = config.get('provider', 'unknown')
        self.model = config.get('model', 'whisper-1')
        self.base_url = config.get('base_url')
        self.api_key = config.get('api_key')
    
    def transcribe(self, audio_file: Path, language: Optional[str] = None, prompt: Optional[str] = None) -> str:
        """Transcribe audio file to text"""
        raise NotImplementedError
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats"""
        return ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']

class OpenAIWhisperProvider(WhisperProvider):
    """OpenAI Whisper API provider"""
    
    def transcribe(self, audio_file: Path, language: Optional[str] = None, prompt: Optional[str] = None) -> str:
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            with open(audio_file, 'rb') as f:
                transcript = client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                    language=language,
                    prompt=prompt,
                    response_format="text"
                )
            
            return transcript
            
        except Exception as e:
            raise Exception(f"OpenAI Whisper transcription failed: {str(e)}")

class DeepgramWhisperProvider(WhisperProvider):
    """Deepgram API provider"""
    
    def transcribe(self, audio_file: Path, language: Optional[str] = None, prompt: Optional[str] = None) -> str:
        try:
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'audio/wav'
            }
            
            params = {
                'model': self.model,
                'smart_format': 'true',
                'punctuate': 'true'
            }
            
            if language:
                params['language'] = language
            
            with open(audio_file, 'rb') as f:
                response = requests.post(
                    f"{self.base_url}/v1/listen",
                    headers=headers,
                    params=params,
                    data=f.read()
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract transcript from Deepgram response
            transcript = result.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript', '')
            return transcript
            
        except Exception as e:
            raise Exception(f"Deepgram transcription failed: {str(e)}")

class LocalWhisperProvider(WhisperProvider):
    """Local Whisper model provider (placeholder)"""
    
    def transcribe(self, audio_file: Path, language: Optional[str] = None, prompt: Optional[str] = None) -> str:
        # This would integrate with local Whisper models
        # For now, return a placeholder
        raise NotImplementedError("Local Whisper provider not yet implemented")

def get_whisper_provider(config: Config, model_name: Optional[str] = None) -> WhisperProvider:
    """Get appropriate Whisper provider based on configuration"""
    
    if model_name:
        model_config = config.get_model_config('whisper', model_name)
    else:
        model_config = config.get_model_config('whisper', config.default_whisper_model)
    
    if not model_config:
        raise ValueError(f"Whisper model '{model_name or config.default_whisper_model}' not found")
    
    provider = model_config.get('provider', '').lower()
    
    if provider in ['openai', 'azure']:
        return OpenAIWhisperProvider(model_config)
    elif provider == 'deepgram':
        return DeepgramWhisperProvider(model_config)
    elif provider == 'local':
        return LocalWhisperProvider(model_config)
    else:
        raise ValueError(f"Unsupported Whisper provider: {provider}")

@app.command()
def transcribe(
    audio_file: Optional[Path] = typer.Argument(None, help="Audio file to transcribe"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Whisper model to use"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output markdown file path"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Audio language (e.g., 'en', 'es', 'fr')"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Optional prompt to guide transcription"),
    mic: bool = typer.Option(False, "--mic", help="Record from microphone instead of file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Transcribe audio to text using Whisper models"""
    
    try:
        config = Config()
        
        # Handle microphone recording
        if mic:
            if audio_file:
                console.print("[yellow]Warning: --mic flag provided with audio file. Using microphone.[/yellow]")
            
            console.print("[green]🎤 Recording from microphone...[/green]")
            console.print("[blue]Press SPACE to stop recording[/blue]")
            
            # Record audio
            temp_dir = Path(tempfile.gettempdir())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recorded_file = temp_dir / f"audai_recording_{timestamp}.wav"
            
            try:
                record_audio_with_key(str(recorded_file), stop_key='space')
                audio_file = recorded_file
                console.print(f"[green]✓ Recording saved to: {recorded_file}[/green]")
            except Exception as e:
                console.print(f"[red]✗ Recording failed: {str(e)}[/red]")
                raise typer.Exit(1)
        
        # Validate audio file
        if not audio_file or not audio_file.exists():
            console.print("[red]✗ Audio file not found or not provided[/red]")
            raise typer.Exit(1)
        
        if not validate_audio_file(audio_file):
            console.print(f"[red]✗ Invalid audio file: {audio_file}[/red]")
            raise typer.Exit(1)
        
        # Get Whisper provider
        try:
            provider = get_whisper_provider(config, model)
        except Exception as e:
            console.print(f"[red]✗ Provider error: {str(e)}[/red]")
            raise typer.Exit(1)
        
        # Display file info
        if verbose:
            duration = get_audio_duration(audio_file)
            file_size = audio_file.stat().st_size / (1024 * 1024)  # MB
            
            info_table = Table(title="Audio File Information")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="white")
            info_table.add_row("File", str(audio_file))
            info_table.add_row("Size", f"{file_size:.2f} MB")
            info_table.add_row("Duration", f"{duration:.2f} seconds")
            info_table.add_row("Model", f"{provider.name} ({provider.model})")
            
            console.print(info_table)
        
        # Transcribe audio
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Transcribing audio...", total=None)
            
            try:
                transcript = provider.transcribe(audio_file, language, prompt)
            except Exception as e:
                console.print(f"[red]✗ Transcription failed: {str(e)}[/red]")
                raise typer.Exit(1)
        
        if not transcript or not transcript.strip():
            console.print("[yellow]⚠ No transcript generated (empty result)[/yellow]")
            return
        
        # Determine output file
        if not output:
            output_dir = Path(config.output_dir)
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = output_dir / f"transcript_{timestamp}.md"
        
        # Create markdown content
        markdown_content = f"# Audio Transcription\n\n"
        markdown_content += f"**File:** {audio_file.name}\n\n"
        markdown_content += f"**Model:** {provider.name} ({provider.model})\n\n"
        markdown_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if language:
            markdown_content += f"**Language:** {language}\n\n"
        
        if prompt:
            markdown_content += f"**Prompt:** {prompt}\n\n"
        
        markdown_content += f"## Transcript\n\n{transcript}\n"
        
        # Save transcript
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(markdown_content, encoding='utf-8')
            
            console.print(Panel(
                f"[green]✓ Transcription completed![/green]\n\n"
                f"[blue]Output:[/blue] {output}\n"
                f"[blue]Length:[/blue] {len(transcript)} characters",
                title="Success",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]✗ Failed to save transcript: {str(e)}[/red]")
            raise typer.Exit(1)
        
        # Clean up temporary recording file
        if mic and recorded_file and recorded_file.exists():
            try:
                recorded_file.unlink()
            except:
                pass  # Ignore cleanup errors
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Unexpected error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

@app.command()
def list_models():
    """List available Whisper models"""
    
    try:
        config = Config()
        models = config.list_models('whisper')
        
        if not models:
            console.print("[yellow]No Whisper models configured[/yellow]")
            return
        
        table = Table(title="Available Whisper Models")
        table.add_column("Name", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="white")
        table.add_column("Base URL", style="blue")
        table.add_column("Default", style="yellow")
        
        for model_name in models:
            model_config = config.get_model_config('whisper', model_name)
            is_default = "✓" if model_name == config.default_whisper_model else ""
            
            table.add_row(
                model_name,
                model_config.get('provider', 'Unknown'),
                model_config.get('model', 'Unknown'),
                model_config.get('base_url', 'Default'),
                is_default
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Error listing models: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def test_whisper(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Test Whisper transcription with a sample audio file"""
    
    try:
        config = Config()
        
        # Get provider
        try:
            provider = get_whisper_provider(config, model)
        except Exception as e:
            console.print(f"[red]✗ Provider error: {str(e)}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[blue]Testing Whisper model: {provider.name} ({provider.model})[/blue]")
        
        # For testing, we would need a sample audio file
        # This is a placeholder for the test functionality
        console.print("[yellow]Test functionality requires a sample audio file[/yellow]")
        console.print("[blue]Use 'audai whisper transcribe --help' for usage information[/blue]")
        
    except Exception as e:
        console.print(f"[red]✗ Test failed: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

if __name__ == "__main__":
    app()