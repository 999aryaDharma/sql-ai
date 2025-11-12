"""Switch and workspace management commands."""
import typer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import (
    print_success, print_error, print_info, print_workspaces, confirm, console
)


def switch_workspace(workspace: str) -> None:
    """Switch to a different workspace."""
    workspace_manager = WorkspaceManager()
    
    try:
        workspace_manager.switch_workspace(workspace)
        print_success(f"Switched to workspace '{workspace}'")
        console.print()
        print_info("You can now generate queries with:")
        console.print(f"  [cyan]sqlgenx generate \"your query here\"[/cyan]")
    except ValueError as e:
        print_error(str(e))
        console.print()
        print_info("See available workspaces with:")
        console.print("  [cyan]sqlgenx list[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to switch workspace: {str(e)}")
        raise typer.Exit(1)


def list_all_workspaces() -> None:
    """List all available workspaces."""
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspaces = workspace_manager.list_workspaces()
    
    if not workspaces:
        console.print()
        print_info("No workspaces found")
        console.print()
        print_info("Create your first workspace with:")
        console.print("  [cyan]sqlgenx load <schema-file.sql>[/cyan]")
        console.print()
        console.print("Example:")
        console.print("  [cyan]sqlgenx load database.sql --name mydb --dbms postgresql[/cyan]")
        return
    
    console.print()
    current = config_manager.get_current_workspace()
    print_workspaces(workspaces, current)
    
    console.print()
    print_info(f"Total workspaces: {len(workspaces)}")
    console.print()
    print_info("Switch workspace with:")
    console.print("  [cyan]sqlgenx use <workspace-name>[/cyan]")


def delete_workspace_cmd(workspace: str, force: bool) -> None:
    """Delete a workspace."""
    workspace_manager = WorkspaceManager()
    
    # Check if workspace exists
    workspace_dir = workspace_manager.get_workspace(workspace)
    if not workspace_dir:
        print_error(f"Workspace '{workspace}' not found")
        console.print()
        print_info("See available workspaces with:")
        console.print("  [cyan]sqlgenx list[/cyan]")
        raise typer.Exit(1)
    
    # Get workspace info
    meta = workspace_manager._load_meta(workspace_dir)
    
    # Confirm deletion
    if not force:
        console.print()
        if meta:
            console.print(f"[bold]Workspace:[/bold] {meta.name}")
            console.print(f"[bold]DBMS:[/bold] {meta.dbms_type}")
            console.print(f"[bold]Created:[/bold] {meta.created_at.split('T')[0]}")
            if meta.description:
                console.print(f"[bold]Description:[/bold] {meta.description}")
            console.print()
        
        if not confirm(f"⚠️  Delete workspace '{workspace}' and all its data?", default=False):
            print_info("Deletion cancelled")
            return
    
    try:
        was_active_deleted = workspace_manager.delete_workspace(workspace)
        console.print()
        print_success(f"Workspace '{workspace}' deleted successfully")
        if was_active_deleted:
            print_info(f"Workspace aktif telah dihapus. Workspace saat ini diatur ulang ke '{workspace_manager.config_manager.default_workspace}'.")
    except ValueError as e:
        console.print()
        print_error(str(e))
        # ValueError "Cannot delete the currently active workspace" tidak akan lagi terjadi
        # karena logika penanganan telah dipindahkan ke WorkspaceManager.
        # ValueErrors lainnya (misalnya, workspace tidak ditemukan) akan tetap dicetak.
        raise typer.Exit(1)
    except Exception as e:
        console.print()
        print_error(f"Failed to delete workspace: {str(e)}")
        raise typer.Exit(1)