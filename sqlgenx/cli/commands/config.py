"""Config command for SQLGenX CLI."""
from typing import Optional
import typer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rich.table import Table
from rich.panel import Panel
from sqlgenx.utils.workspace_manager import ConfigManager, WorkspaceManager
from sqlgenx.utils.rich_helpers import print_success, print_error, print_info, console
from sqlgenx.core.vector_store import VectorStore


def config_cmd(key: Optional[str], value: Optional[str], show: bool) -> None:
    """Configure SQLGenX settings."""
    config_manager = ConfigManager()
    
    if show:
        # Show current configuration
        console.print()
        table = Table(title="SQLGenX Configuration", show_header=True, header_style="bold violet")
        table.add_column("Setting", style="cyan", width=25)
        table.add_column("Value", style="cyan2")
        
        # Show key settings
        current_ws = config_manager.get_current_workspace()
        sqlgenx_home = str(config_manager.sqlgenx_home)
        api_key = config_manager.gemini_api_key
        api_key_display = f"{api_key[:10]}..." if api_key else "[red]not set[/red]"
        
        table.add_row("Current Workspace", current_ws)
        table.add_row("SQLGenX Home", sqlgenx_home)
        table.add_row("Gemini API Key", api_key_display)
        
        console.print(table)
        console.print()
        
        if not api_key:
            print_info("Configure your API key:")
            console.print("  1. Environment variable:")
            console.print("     [cyan]export GEMINI_API_KEY='your-key'[/cyan]")
            console.print()
            console.print("  2. Create a .env file:")
            console.print("     [cyan]GEMINI_API_KEY=your-key[/cyan]")
            console.print()
            console.print("  Get your key: [link]https://makersuite.google.com/app/apikey[/link]")
        else:
            print_success("API key is configured ✓")
        
        return
    
    if not key or not value:
        print_error("Please provide both key and value")
        console.print()
        print_info("Usage:")
        console.print("  [cyan]sqlgenx config <key> <value>[/cyan]")
        console.print()
        print_info("Or show current configuration:")
        console.print("  [cyan]sqlgenx config --show[/cyan]")
        raise typer.Exit(1)
    
    # Set configuration
    try:
        config_manager.save_global_config(key, value)
        print_success(f"Configuration updated: {key} = {value}")
    except Exception as e:
        print_error(f"Failed to save configuration: {str(e)}")
        raise typer.Exit(1)


def info_cmd(workspace: Optional[str]) -> None:
    """Show information about a workspace."""
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    # Determine which workspace
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        console.print()
        print_info("See available workspaces:")
        console.print("  [cyan]sqlgenx list[/cyan]")
        raise typer.Exit(1)
    
    # Load metadata
    meta = workspace_manager._load_meta(workspace_dir)
    if not meta:
        print_error("Workspace metadata not found")
        print_info("Workspace may be corrupted")
        raise typer.Exit(1)
    
    # Get tables from vector store
    try:
        vector_store = VectorStore(workspace_dir)
        tables = vector_store.get_all_tables()
    except Exception:
        tables = []
    
    # Display information
    console.print()
    
    info_lines = []
    info_lines.append(f"[bold cyan]Workspace:[/bold cyan] {meta.name}")
    info_lines.append(f"[bold]DBMS Type:[/bold] {meta.dbms_type}")
    info_lines.append(f"[bold]Created:[/bold] {meta.created_at.split('T')[0]}")
    info_lines.append(f"[bold]Last Used:[/bold] {meta.last_used.split('T')[0]}")
    info_lines.append(f"[bold]Schema Path:[/bold] {meta.schema_path}")
    info_lines.append(f"[bold]Tables:[/bold] {len(tables)}")
    
    if meta.description:
        info_lines.append(f"[bold]Description:[/bold] {meta.description}")
    
    info_text = "\n".join(info_lines)
    
    panel = Panel(info_text, title=f"📊 Workspace Information", border_style="cyan")
    console.print(panel)
    
    if tables:
        console.print()
        console.print("[bold]Available Tables:[/bold]")
        
        # Show tables in columns
        if len(tables) <= 20:
            for table in tables:
                console.print(f"  • [cyan]{table}[/cyan]")
        else:
            # Show first 20 and indicate more
            for table in tables[:20]:
                console.print(f"  • [cyan]{table}[/cyan]")
            console.print(f"  [dim]... and {len(tables) - 20} more[/dim]")
    
    console.print()
    print_info("Generate a query with:")
    console.print(f"  [cyan]sqlgenx generate \"your query here\"[/cyan]")