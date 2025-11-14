"""Cache management commands."""
from typing import Optional
import typer
from rich.table import Table

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import print_success, print_info, console
from sqlgenx.core.cache.manager import CacheManager


def cache_cmd(clear: bool, stats: bool, workspace: Optional[str]) -> None:
    """Manage cache for a workspace."""
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        console.print(f"[red]Workspace '{workspace_name}' not found[/red]")
        raise typer.Exit(1)
    
    cache_manager = CacheManager(workspace_dir)
    
    if clear:
        cache_manager.clear_all()
        print_success("Cache cleared successfully")
        return
    
    if stats:
        stats = cache_manager.get_stats()
        
        table = Table(title=f"Cache Statistics - {workspace_name}")
        table.add_column("Type", style="cyan")
        table.add_column("Entries", style="yellow")
        
        table.add_row("SQL Generation", str(stats['sql_entries']))
        table.add_row("Validation", str(stats['validation_entries']))
        table.add_row("Analysis", str(stats['analysis_entries']))
        table.add_row("Optimization", str(stats['optimization_entries']))
        table.add_row("[bold]Total[/bold]", f"[bold]{stats['total_entries']}[/bold]")
        
        console.print(table)
        console.print(f"\n[dim]Cache directory: {stats['cache_dir']}[/dim]")
        return
    
    print_info("Use --clear to clear cache or --stats to show statistics")