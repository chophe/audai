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
import json

from ..config import Config
from ..audio_utils import (
    record_audio_with_key, 
    play_audio, 
    validate_audio_file, 
    get_audio_duration, 
    generate_audio_filename,
    AudioProgressCallback
)

console = Console()
app = typer.Typer(help="Audio-to-audio conversation using AI models")

class TalkProvider:
    """Base class for audio-to-audio conversation providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'Unknown')
        self.provider = config.get('provider', 'unknown')
        self.model = config.get('model', 'gpt-4-audio-preview')
        self.voice = config.get('voice', 'alloy')
        self.base_url = config.get('base_url')
        self.api_key = config.get('api_key')
    
    def process_audio(self, audio_file: Path, system_prompt: Optional[str] = None) -> Path:
        """Process audio input and return audio response"""
        raise NotImplementedError
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats"""
        return ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']

class OpenAITalkProvider(TalkProvider):
    """OpenAI audio-to-audio provider (GPT-4 Audio)"""
    
    def process_audio(self, audio_file: Path, system_prompt: Optional[str] = None) -> Path:
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Prepare messages for audio conversation
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add audio input
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": "wav"  # Adjust based on file format
                        }
                    }
                ]
            })
            
            # Make API call with audio response
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["text", "audio"],
                audio={
                    "voice": self.voice,
                    "format": "wav"
                }
            )
            
            # Extract audio response
            if hasattr(response.choices[0].message, 'audio') and response.choices[0].message.audio:
                audio_data = response.choices[0].message.audio.data
                
                # Save response audio
                output_file = Path(tempfile.gettempdir()) / generate_audio_filename("talk_response", "wav")
                output_file.write_bytes(audio_data)
                
                return output_file
            else:
                raise Exception("No audio response received from the model")
            
        except Exception as e:
            raise Exception(f"OpenAI Talk processing failed: {str(e)}")

class ReplicateTalkProvider(TalkProvider):
    """Replicate audio-to-audio provider"""
    
    def process_audio(self, audio_file: Path, system_prompt: Optional[str] = None) -> Path:
        try:
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Prepare input data
            with open(audio_file, 'rb') as f:
                import base64
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            data = {
                "version": self.model,
                "input": {
                    "audio": f"data:audio/wav;base64,{audio_b64}",
                    "system_prompt": system_prompt or "You are a helpful assistant."
                }
            }
            
            # Make prediction request
            response = requests.post(
                f"{self.base_url}/v1/predictions",
                headers=headers,
                json=data
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Poll for completion (simplified)
            prediction_id = result.get('id')
            if prediction_id:
                # In a real implementation, you'd poll the status endpoint
                # For now, we'll assume immediate response
                pass
            
            # Extract audio URL and download
            audio_url = result.get('output', {}).get('audio_url')
            if audio_url:
                audio_response = requests.get(audio_url)
                audio_response.raise_for_status()
                
                output_file = Path(tempfile.gettempdir()) / generate_audio_filename("talk_response", "wav")
                output_file.write_bytes(audio_response.content)
                
                return output_file
            else:
                raise Exception("No audio URL in response")
            
        except Exception as e:
            raise Exception(f"Replicate Talk processing failed: {str(e)}")

class LocalTalkProvider(TalkProvider):
    """Local audio-to-audio provider (placeholder)"""
    
    def process_audio(self, audio_file: Path, system_prompt: Optional[str] = None) -> Path:
        # This would integrate with local audio-to-audio models
        # For now, return a placeholder
        raise NotImplementedError("Local Talk provider not yet implemented")

def get_talk_provider(config: Config, model_name: Optional[str] = None) -> TalkProvider:
    """Get appropriate Talk provider based on configuration"""
    
    if model_name:
        model_config = config.get_model_config('talk', model_name)
    else:
        model_config = config.get_model_config('talk', config.default_talk_model)
    
    if not model_config:
        raise ValueError(f"Talk model '{model_name or config.default_talk_model}' not found")
    
    provider = model_config.get('provider', '').lower()
    
    if provider in ['openai', 'azure']:
        return OpenAITalkProvider(model_config)
    elif provider == 'replicate':
        return ReplicateTalkProvider(model_config)
    elif provider == 'local':
        return LocalTalkProvider(model_config)
    else:
        raise ValueError(f"Unsupported Talk provider: {provider}")

@app.command()
def conversation(
    audio_file: Optional[Path] = typer.Argument(None, help="Audio file to send"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Talk model to use"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt for the conversation"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save response audio to file"),
    mic: bool = typer.Option(False, "--mic", help="Record from microphone"),
    play: bool = typer.Option(True, "--play/--no-play", help="Play the response audio"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Have an audio conversation with AI models"""
    
    try:
        config = Config()
        
        # Handle microphone recording
        if mic:
            if audio_file:
                console.print("[yellow]Warning: --mic flag provided with audio file. Using microphone.[/yellow]")
            
            console.print("[green]🎤 Recording your message...[/green]")
            console.print("[blue]Press SPACE to stop recording[/blue]")
            
            # Record audio
            temp_dir = Path(tempfile.gettempdir())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recorded_file = temp_dir / f"audai_talk_input_{timestamp}.wav"
            
            try:
                record_audio_with_key(str(recorded_file), stop_key='space')
                audio_file = recorded_file
                console.print(f"[green]✓ Recording saved[/green]")
            except Exception as e:
                console.print(f"[red]✗ Recording failed: {str(e)}[/red]")
                raise typer.Exit(1)
        
        # Validate audio file
        if not audio_file or not audio_file.exists():
            console.print("[red]✗ Audio file not found or not provided[/red]")
            console.print("[blue]Use --mic to record from microphone[/blue]")
            raise typer.Exit(1)
        
        if not validate_audio_file(audio_file):
            console.print(f"[red]✗ Invalid audio file: {audio_file}[/red]")
            raise typer.Exit(1)
        
        # Get Talk provider
        try:
            provider = get_talk_provider(config, model)
        except Exception as e:
            console.print(f"[red]✗ Provider error: {str(e)}[/red]")
            raise typer.Exit(1)
        
        # Display conversation info
        if verbose:
            duration = get_audio_duration(audio_file)
            file_size = audio_file.stat().st_size / (1024 * 1024)  # MB
            
            info_table = Table(title="Conversation Information")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="white")
            info_table.add_row("Input File", str(audio_file))
            info_table.add_row("Size", f"{file_size:.2f} MB")
            info_table.add_row("Duration", f"{duration:.2f} seconds")
            info_table.add_row("Model", f"{provider.name} ({provider.model})")
            info_table.add_row("Voice", provider.voice)
            
            if system_prompt:
                info_table.add_row("System Prompt", system_prompt[:50] + "..." if len(system_prompt) > 50 else system_prompt)
            
            console.print(info_table)
        
        # Process audio conversation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing conversation...", total=None)
            
            try:
                response_audio = provider.process_audio(audio_file, system_prompt)
            except Exception as e:
                console.print(f"[red]✗ Conversation processing failed: {str(e)}[/red]")
                raise typer.Exit(1)
        
        if not response_audio or not response_audio.exists():
            console.print("[yellow]⚠ No audio response generated[/yellow]")
            return
        
        # Save output if requested
        if output:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy response to output location
                import shutil
                shutil.copy2(response_audio, output)
                
                console.print(f"[green]✓ Response saved to: {output}[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to save response: {str(e)}[/red]")
        
        # Play response audio
        if play:
            try:
                console.print("[green]🔊 Playing AI response...[/green]")
                play_audio(response_audio)
                console.print("[green]✓ Playback completed[/green]")
            except Exception as e:
                console.print(f"[red]✗ Playback failed: {str(e)}[/red]")
        
        # Show success summary
        response_duration = get_audio_duration(response_audio)
        
        console.print(Panel(
            f"[green]✓ Conversation completed![/green]\n\n"
            f"[blue]Response Duration:[/blue] {response_duration:.2f} seconds\n"
            f"[blue]Model:[/blue] {provider.name} ({provider.model})\n"
            f"[blue]Voice:[/blue] {provider.voice}",
            title="Success",
            border_style="green"
        ))
        
        # Clean up temporary files
        if mic and 'recorded_file' in locals() and recorded_file.exists():
            try:
                recorded_file.unlink()
            except:
                pass  # Ignore cleanup errors
        
        # Clean up response file if not saved
        if not output:
            try:
                response_audio.unlink()
            except:
                pass  # Ignore cleanup errors
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Conversation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Unexpected error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

@app.command()
def interactive(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Talk model to use"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt for the conversation"),
    save_conversation: bool = typer.Option(False, "--save", help="Save conversation audio files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Start an interactive audio conversation session"""
    
    try:
        config = Config()
        
        # Get Talk provider
        try:
            provider = get_talk_provider(config, model)
        except Exception as e:
            console.print(f"[red]✗ Provider error: {str(e)}[/red]")
            raise typer.Exit(1)
        
        console.print(Panel(
            f"[green]🎙️ Interactive Audio Conversation[/green]\n\n"
            f"[blue]Model:[/blue] {provider.name} ({provider.model})\n"
            f"[blue]Voice:[/blue] {provider.voice}\n\n"
            f"[yellow]Commands:[/yellow]\n"
            f"• Press SPACE to record your message\n"
            f"• Type 'quit' or 'exit' to end conversation\n"
            f"• Press Ctrl+C to cancel",
            title="Interactive Mode",
            border_style="blue"
        ))
        
        conversation_count = 0
        
        while True:
            try:
                # Get user input
                user_input = typer.prompt("\nPress ENTER to start recording (or 'quit' to exit)")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[green]👋 Goodbye![/green]")
                    break
                
                conversation_count += 1
                
                # Record user audio
                console.print("[green]🎤 Recording your message...[/green]")
                console.print("[blue]Press SPACE to stop recording[/blue]")
                
                temp_dir = Path(tempfile.gettempdir())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                input_file = temp_dir / f"audai_interactive_input_{conversation_count}_{timestamp}.wav"
                
                record_audio_with_key(str(input_file), stop_key='space')
                
                # Process conversation
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("AI is thinking...", total=None)
                    response_audio = provider.process_audio(input_file, system_prompt)
                
                # Play response
                console.print("[green]🔊 AI Response:[/green]")
                play_audio(response_audio)
                
                # Save conversation if requested
                if save_conversation:
                    output_dir = Path(config.output_dir) / "conversations" / datetime.now().strftime("%Y%m%d")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save input and response
                    input_save = output_dir / f"input_{conversation_count}_{timestamp}.wav"
                    response_save = output_dir / f"response_{conversation_count}_{timestamp}.wav"
                    
                    import shutil
                    shutil.copy2(input_file, input_save)
                    shutil.copy2(response_audio, response_save)
                    
                    console.print(f"[blue]💾 Conversation saved to: {output_dir}[/blue]")
                
                # Clean up temporary files
                try:
                    input_file.unlink()
                    response_audio.unlink()
                except:
                    pass
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Conversation interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]✗ Error in conversation: {str(e)}[/red]")
                continue
        
    except Exception as e:
        console.print(f"[red]✗ Interactive session failed: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

@app.command()
def list_models():
    """List available Talk models"""
    
    try:
        config = Config()
        models = config.list_models('talk')
        
        if not models:
            console.print("[yellow]No Talk models configured[/yellow]")
            return
        
        table = Table(title="Available Talk Models")
        table.add_column("Name", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="white")
        table.add_column("Voice", style="magenta")
        table.add_column("Base URL", style="blue")
        table.add_column("Default", style="yellow")
        
        for model_name in models:
            model_config = config.get_model_config('talk', model_name)
            is_default = "✓" if model_name == config.default_talk_model else ""
            
            table.add_row(
                model_name,
                model_config.get('provider', 'Unknown'),
                model_config.get('model', 'Unknown'),
                model_config.get('voice', 'Default'),
                model_config.get('base_url', 'Default'),
                is_default
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Error listing models: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def test_talk(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Test Talk functionality with microphone input"""
    
    try:
        config = Config()
        
        # Get provider
        try:
            provider = get_talk_provider(config, model)
        except Exception as e:
            console.print(f"[red]✗ Provider error: {str(e)}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[blue]Testing Talk model: {provider.name} ({provider.model})[/blue]")
        console.print("[green]🎤 Record a test message...[/green]")
        console.print("[blue]Press SPACE to stop recording[/blue]")
        
        # Record test audio
        temp_dir = Path(tempfile.gettempdir())
        test_file = temp_dir / "audai_talk_test.wav"
        
        record_audio_with_key(str(test_file), stop_key='space')
        
        # Test conversation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Testing Talk model...", total=None)
            response_audio = provider.process_audio(test_file, "You are a helpful assistant. Please respond briefly.")
        
        # Play response
        console.print("[green]🔊 Playing test response...[/green]")
        play_audio(response_audio)
        
        console.print("[green]✓ Talk test completed successfully![/green]")
        
        # Clean up
        try:
            test_file.unlink()
            response_audio.unlink()
        except:
            pass
        
    except Exception as e:
        console.print(f"[red]✗ Test failed: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

if __name__ == "__main__":
    app()