"""AudAI - AI Audio CLI for TTS, Whisper, and Audio-to-Audio conversations."""

__version__ = "0.1.0"
__author__ = "AudAI Team"
__description__ = "AI Audio CLI for TTS, Whisper, and Audio-to-Audio conversations"

from .cli import main_cli
from .config import Config

__all__ = ["main_cli", "Config", "__version__"]
