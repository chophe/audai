"""Audio utilities for recording, playback, and file management."""

import os
import time
import wave
import threading
from pathlib import Path
from typing import Optional, Callable
import uuid
from datetime import datetime

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import keyboard
except ImportError:
    keyboard = None

try:
    from pydub import AudioSegment
    from pydub.playback import play
except ImportError:
    AudioSegment = None
    play = None

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

console = Console()

class AudioRecorder:
    """Audio recording utility with microphone input."""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 1, chunk_size: int = 1024):
        """Initialize audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
            chunk_size: Audio chunk size for recording.
        """
        if not pyaudio:
            raise ImportError("pyaudio is required for audio recording. Install with: pip install pyaudio")
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
    
    def start_recording(self) -> bool:
        """Start recording audio.
        
        Returns:
            True if recording started successfully, False otherwise.
        """
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.recording = True
            self.frames = []
            return True
        except Exception as e:
            console.print(f"[red]Error starting recording: {e}[/red]")
            return False
    
    def stop_recording(self) -> bool:
        """Stop recording audio.
        
        Returns:
            True if recording stopped successfully, False otherwise.
        """
        try:
            self.recording = False
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
            return True
        except Exception as e:
            console.print(f"[red]Error stopping recording: {e}[/red]")
            return False
    
    def record_chunk(self):
        """Record a single audio chunk."""
        if self.recording and hasattr(self, 'stream'):
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                console.print(f"[red]Error recording chunk: {e}[/red]")
    
    def save_recording(self, filename: str) -> bool:
        """Save recorded audio to file.
        
        Args:
            filename: Output filename.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            return True
        except Exception as e:
            console.print(f"[red]Error saving recording: {e}[/red]")
            return False
    
    def cleanup(self):
        """Clean up audio resources."""
        if hasattr(self, 'stream'):
            try:
                self.stream.close()
            except:
                pass
        try:
            self.audio.terminate()
        except:
            pass

def record_audio_with_key(output_path: str, stop_key: str = 'space', timeout: int = 30) -> bool:
    """Record audio until a key is pressed or timeout.
    
    Args:
        output_path: Path to save the recorded audio.
        stop_key: Key to stop recording (default: 'space').
        timeout: Maximum recording time in seconds.
        
    Returns:
        True if recording was successful, False otherwise.
    """
    if not keyboard:
        console.print("[red]keyboard library is required for key-based recording. Install with: pip install keyboard[/red]")
        return False
    
    recorder = AudioRecorder()
    
    if not recorder.start_recording():
        return False
    
    # Recording status display
    recording_text = Text()
    recording_text.append("🎤 Recording... ", style="red bold")
    recording_text.append(f"Press [{stop_key.upper()}] to stop or wait {timeout}s for timeout", style="dim")
    
    start_time = time.time()
    stop_pressed = False
    
    def on_key_press():
        nonlocal stop_pressed
        stop_pressed = True
    
    # Set up key listener
    keyboard.on_press_key(stop_key, lambda _: on_key_press())
    
    try:
        with Live(Panel(recording_text, title="Audio Recording", border_style="red"), refresh_per_second=4) as live:
            while recorder.recording and not stop_pressed:
                recorder.record_chunk()
                elapsed = time.time() - start_time
                
                # Update display
                recording_text = Text()
                recording_text.append("🎤 Recording... ", style="red bold")
                recording_text.append(f"[{elapsed:.1f}s] ", style="yellow")
                recording_text.append(f"Press [{stop_key.upper()}] to stop", style="dim")
                
                live.update(Panel(recording_text, title="Audio Recording", border_style="red"))
                
                # Check timeout
                if elapsed >= timeout:
                    console.print(f"[yellow]Recording stopped due to {timeout}s timeout[/yellow]")
                    break
                
                time.sleep(0.1)
    
    finally:
        keyboard.unhook_all()
        recorder.stop_recording()
    
    # Save the recording
    success = recorder.save_recording(output_path)
    recorder.cleanup()
    
    if success:
        console.print(f"[green]✓ Recording saved to: {output_path}[/green]")
    
    return success

def play_audio(file_path: str) -> bool:
    """Play audio file.
    
    Args:
        file_path: Path to the audio file.
        
    Returns:
        True if playback was successful, False otherwise.
    """
    if not os.path.exists(file_path):
        console.print(f"[red]Audio file not found: {file_path}[/red]")
        return False
    
    try:
        if AudioSegment and play:
            # Use pydub for playback
            audio = AudioSegment.from_file(file_path)
            console.print(f"[green]🔊 Playing: {Path(file_path).name}[/green]")
            play(audio)
            return True
        else:
            # Fallback to system command
            import subprocess
            import platform
            
            system = platform.system().lower()
            if system == 'windows':
                os.startfile(file_path)
            elif system == 'darwin':  # macOS
                subprocess.run(['open', file_path])
            elif system == 'linux':
                subprocess.run(['xdg-open', file_path])
            else:
                console.print(f"[yellow]Cannot play audio on {system}. File saved at: {file_path}[/yellow]")
                return False
            
            console.print(f"[green]🔊 Playing: {Path(file_path).name}[/green]")
            return True
            
    except Exception as e:
        console.print(f"[red]Error playing audio: {e}[/red]")
        return False

def generate_audio_filename(output_dir: Path, prefix: str = "audio", extension: str = "mp3") -> str:
    """Generate a unique audio filename.
    
    Args:
        output_dir: Output directory.
        prefix: Filename prefix.
        extension: File extension.
        
    Returns:
        Full path to the generated filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}.{extension}"
    return str(output_dir / filename)

def convert_audio_format(input_path: str, output_path: str, target_format: str = "mp3") -> bool:
    """Convert audio file to different format.
    
    Args:
        input_path: Input audio file path.
        output_path: Output audio file path.
        target_format: Target audio format.
        
    Returns:
        True if conversion was successful, False otherwise.
    """
    if not AudioSegment:
        console.print("[red]pydub is required for audio conversion. Install with: pip install pydub[/red]")
        return False
    
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format=target_format)
        console.print(f"[green]✓ Converted {input_path} to {output_path}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error converting audio: {e}[/red]")
        return False

def get_audio_duration(file_path: str) -> Optional[float]:
    """Get audio file duration in seconds.
    
    Args:
        file_path: Path to the audio file.
        
    Returns:
        Duration in seconds, or None if error.
    """
    if not AudioSegment:
        return None
    
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception:
        return None

def validate_audio_file(file_path: str) -> bool:
    """Validate if file is a valid audio file.
    
    Args:
        file_path: Path to the audio file.
        
    Returns:
        True if valid audio file, False otherwise.
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    valid_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    if Path(file_path).suffix.lower() not in valid_extensions:
        return False
    
    # Try to load with pydub if available
    if AudioSegment:
        try:
            AudioSegment.from_file(file_path)
            return True
        except Exception:
            return False
    
    # Basic file size check
    return os.path.getsize(file_path) > 0

class AudioProgressCallback:
    """Callback class for showing audio processing progress."""
    
    def __init__(self, description: str = "Processing audio"):
        self.description = description
        self.progress = None
        self.task = None
    
    def start(self, total: Optional[int] = None):
        """Start progress display."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        self.progress.start()
        self.task = self.progress.add_task(self.description, total=total)
    
    def update(self, advance: int = 1):
        """Update progress."""
        if self.progress and self.task:
            self.progress.update(self.task, advance=advance)
    
    def stop(self):
        """Stop progress display."""
        if self.progress:
            self.progress.stop()