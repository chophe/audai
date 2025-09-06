"""Main CLI application for AudAI."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from .config import Config
from .commands import tts
from .commands import whisper
from .commands import talk

# Initialize Typer app
app = typer.Typer(
    name="audai",
    help="🎵 AudAI - AI Audio CLI for TTS, Whisper, and Audio-to-Audio conversations",
    rich_markup_mode="rich",
    no_args_is_help=True
)

console = Console()

@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version information"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output")
):
    """🎵 AudAI - AI Audio CLI for TTS, Whisper, and Audio-to-Audio conversations.
    
    A powerful command-line tool for working with AI audio services including:
    • Text-to-Speech (TTS) with multiple providers
    • Speech-to-Text (Whisper) with transcription
    • Audio-to-Audio conversations
    """
    if version:
        show_version()
        raise typer.Exit()
    
    # Store config info in context
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config_file
    ctx.obj['verbose'] = verbose
    
    # Set verbose mode
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")

def show_version():
    """Display version information."""
    version_text = Text()
    version_text.append("AudAI ", style="bold blue")
    version_text.append("v0.1.0", style="green")
    version_text.append("\n\nAI Audio CLI Tool", style="dim")
    version_text.append("\nSupports TTS, Whisper, and Audio-to-Audio conversations", style="dim")
    
    console.print(Panel(
        version_text,
        title="Version Information",
        border_style="blue"
    ))

@app.command("config")
def show_config(
    validate: bool = typer.Option(False, "--validate", help="Validate configuration")
):
    """📋 Show current configuration settings."""
    try:
        config = Config()
        if validate:
            if config.validate_config():
                console.print("[green]✓ Configuration is valid[/green]")
            else:
                console.print("[red]✗ Configuration has issues[/red]")
                raise typer.Exit(1)
        else:
            config.show_config()
    except Exception as e:
        console.print(f"[red]✗ Configuration error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("models")
def list_models(
    model_type: Optional[str] = typer.Argument(None, help="Model type (tts, whisper, talk)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed model information")
):
    """📝 List available models."""
    if model_type and model_type not in ['tts', 'whisper', 'talk']:
        console.print(f"[red]Invalid model type: {model_type}. Use: tts, whisper, or talk[/red]")
        raise typer.Exit(1)
    
    if model_type:
        show_models_for_type(model_type, detailed)
    else:
        show_all_models(detailed)

def show_models_for_type(model_type: str, detailed: bool = False):
    """Show models for a specific type."""
    try:
        config = Config()
        models = config.list_models(model_type)
        
        if not models:
            console.print(f"[yellow]No {model_type} models configured[/yellow]")
            return
        
        table = Table(title=f"{model_type.upper()} Models")
        table.add_column("Name", style="cyan")
        
        if detailed:
            table.add_column("Provider", style="green")
            table.add_column("Model ID", style="yellow")
            table.add_column("Base URL", style="dim")
        
        for model_name in models:
            model_config = config.get_model_config(model_type, model_name)
            
            # Mark default model
            name_display = model_name
            default_model = getattr(config, f"default_{model_type}_model")
            if model_name == default_model:
                name_display = f"[bold]{model_name}[/bold] [dim](default)[/dim]"
            
            if detailed:
                table.add_row(
                    name_display,
                    model_config.get('provider', 'Unknown'),
                    model_config.get('model', 'Unknown'),
                    model_config.get('base_url', 'N/A')
                )
            else:
                table.add_row(name_display)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗ Error loading models: {str(e)}[/red]")
        raise typer.Exit(1)

def show_all_models(detailed: bool = False):
    """Show all available models."""
    for model_type in ['tts', 'whisper', 'talk']:
        show_models_for_type(model_type, detailed)
        console.print()  # Add spacing

@app.command("init")
def init_config(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing .env file")
):
    """🚀 Initialize configuration by copying .env.template to .env."""
    env_file = Path(".env")
    template_file = Path(".env.template")
    
    if env_file.exists() and not force:
        console.print(f"[yellow].env file already exists. Use --force to overwrite.[/yellow]")
        raise typer.Exit(1)
    
    if not template_file.exists():
        console.print(f"[red].env.template file not found. Please ensure it exists in the current directory.[/red]")
        raise typer.Exit(1)
    
    try:
        # Copy template to .env
        with open(template_file, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        
        console.print(Panel(
            "[green]✓ Configuration initialized successfully![/green]\n\n"
            "[bold]Next steps:[/bold]\n"
            "1. Edit .env file with your API keys and settings\n"
            "2. Run [cyan]audai config --validate[/cyan] to check your configuration\n"
            "3. Run [cyan]audai models[/cyan] to see available models",
            title="Initialization Complete",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error initializing configuration: {e}[/red]")
        raise typer.Exit(1)

# Add command groups
app.add_typer(tts.tts_command, name="tts", help="🗣️  Text-to-Speech commands")
app.add_typer(whisper.app, name="whisper", help="👂 Speech-to-Text commands")
app.add_typer(talk.app, name="talk", help="💬 Audio-to-Audio conversation commands")

@app.command("doctor")
def doctor():
    """🩺 Run diagnostic checks on the system."""
    console.print(Panel(
        "[bold blue]AudAI System Diagnostics[/bold blue]",
        border_style="blue"
    ))
    
    checks = []
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python Version", python_version, "✓" if sys.version_info >= (3, 8) else "✗"))
    
    # Check dependencies
    dependencies = [
        ("typer", "typer"),
        ("rich", "rich"),
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
        ("pyaudio", "pyaudio"),
        ("pydub", "pydub"),
        ("keyboard", "keyboard"),
        ("requests", "requests")
    ]
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            checks.append((f"{dep_name} package", "Installed", "✓"))
        except ImportError:
            checks.append((f"{dep_name} package", "Missing", "✗"))
    
    # Check configuration
    try:
        config = Config()
        config_valid = config.validate_config()
        checks.append(("Configuration", "Valid" if config_valid else "Invalid", "✓" if config_valid else "✗"))
        
        # Check output directory
        output_dir_exists = config.output_dir.exists()
        output_dir_writable = output_dir_exists and config.output_dir.is_dir()
        checks.append(("Output Directory", str(config.output_dir), "✓" if output_dir_writable else "✗"))
    except Exception as e:
        checks.append(("Configuration", f"Error: {str(e)}", "✗"))
        checks.append(("Output Directory", "N/A", "✗"))
    
    # Display results
    table = Table(title="Diagnostic Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Result", justify="center")
    
    for component, status, result in checks:
        style = "green" if result == "✓" else "red"
        table.add_row(component, status, f"[{style}]{result}[/{style}]")
    
    console.print(table)
    
    # Summary
    passed = sum(1 for _, _, result in checks if result == "✓")
    total = len(checks)
    
    if passed == total:
        console.print("\n[green]🎉 All checks passed! AudAI is ready to use.[/green]")
    else:
        console.print(f"\n[yellow]⚠️  {total - passed} issues found. Please address them before using AudAI.[/yellow]")
        
        # Provide helpful suggestions
        console.print("\n[bold]Suggestions:[/bold]")
        console.print("• Install missing packages: [cyan]pip install -r requirements.txt[/cyan]")
        console.print("• Initialize configuration: [cyan]audai init[/cyan]")
        console.print("• Validate configuration: [cyan]audai config --validate[/cyan]")

def main_cli():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main_cli()