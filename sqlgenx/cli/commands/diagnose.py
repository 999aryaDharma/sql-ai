"""Diagnostic commands for troubleshooting."""
from typing import Optional
import typer
import sys
import os
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlgenx.core.schema_loader import SchemaLoader
from sqlgenx.core.vector_store import VectorStore
from sqlgenx.utils.rich_helpers import confirm
from sqlgenx.core.schema_loader import SchemaLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import print_success, print_error, print_info, console, print_warning
from sqlgenx.utils.helpers import normalize_dialect
from rich.table import Table
from rich.panel import Panel


def diagnose_workspace(workspace: Optional[str] = None) -> None:
    """Diagnose workspace issues and show detailed status."""
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    console.print()
    console.print(f"[bold cyan]🔍 Diagnosing workspace: {workspace_name}[/bold cyan]")
    console.print()
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        console.print()
        print_info("Available workspaces:")
        workspaces = workspace_manager.list_workspaces()
        for ws in workspaces:
            console.print(f"  • {ws.name}")
        raise typer.Exit(1)
    
    # Check workspace directory structure
    checks = []
    
    # 1. Workspace directory exists
    checks.append({
        "check": "Workspace directory",
        "status": "✓" if workspace_dir.exists() else "✗",
        "path": str(workspace_dir),
        "ok": workspace_dir.exists()
    })
    
    # 2. Meta file
    meta_file = workspace_dir / "meta.json"
    checks.append({
        "check": "Metadata file",
        "status": "✓" if meta_file.exists() else "✗",
        "path": str(meta_file),
        "ok": meta_file.exists()
    })
    
    # 3. Schema SQL file
    schema_sql = workspace_dir / "schema.sql"
    checks.append({
        "check": "Schema SQL file",
        "status": "✓" if schema_sql.exists() else "✗",
        "path": str(schema_sql),
        "ok": schema_sql.exists()
    })
    
    if schema_sql.exists():
        size = schema_sql.stat().st_size
        checks[-1]["details"] = f"{size:,} bytes"
        
        # Check if empty
        if size == 0:
            checks[-1]["status"] = "⚠"
            checks[-1]["ok"] = False
            checks[-1]["details"] += " (EMPTY!)"
    
    # 4. Embeddings directory
    embeddings_dir = workspace_dir / "embeddings"
    checks.append({
        "check": "Embeddings directory",
        "status": "✓" if embeddings_dir.exists() else "✗",
        "path": str(embeddings_dir),
        "ok": embeddings_dir.exists()
    })
    
    if embeddings_dir.exists():
        files = list(embeddings_dir.iterdir())
        checks[-1]["details"] = f"{len(files)} files"
        
        if len(files) == 0:
            checks[-1]["status"] = "⚠"
            checks[-1]["details"] += " (EMPTY!)"
    
    # 5. History directory
    history_dir = workspace_dir / "history"
    checks.append({
        "check": "History directory",
        "status": "✓" if history_dir.exists() else "✗",
        "path": str(history_dir),
        "ok": history_dir.exists()
    })
    
    # 6. Connections file
    connections_file = workspace_dir / "connections.json"
    has_connection = connections_file.exists()
    checks.append({
        "check": "Database connection",
        "status": "✓" if has_connection else "-",
        "path": str(connections_file) if has_connection else "Not configured",
        "ok": True  # Optional, not an error
    })
    
    # Display checks table
    table = Table(title="Workspace Health Check", show_header=True, header_style="bold violet")
    table.add_column("Status", style="cyan", width=8)
    table.add_column("Check", style="yellow")
    table.add_column("Details", style="dim")
    
    for check in checks:
        style = "cyan2" if check["ok"] else "red"
        details = check.get("details", "")
        if not check["ok"] and not details:
            details = check["path"]
        
        table.add_row(
            f"[{style}]{check['status']}[/{style}]",
            check["check"],
            details
        )
    
    console.print(table)
    console.print()
    
    # Overall status
    all_ok = all(c["ok"] for c in checks[:5])  # Exclude optional connection check
    
    if all_ok:
        print_success("Workspace is healthy!")
    else:
        print_warning("Workspace has issues!")
        console.print()
        print_info("Suggested fixes:")
        
        if not schema_sql.exists() or (schema_sql.exists() and schema_sql.stat().st_size == 0):
            console.print("  • Schema file missing or empty")
            console.print("    Fix: Reconnect to database or reload schema")
            console.print("    [cyan]sqlgenx connect --url \"...\" --name " + workspace_name + "[/cyan]")
        
        if not embeddings_dir.exists() or (embeddings_dir.exists() and len(list(embeddings_dir.iterdir())) == 0):
            console.print("  • Embeddings missing")
            console.print("    Fix: Rebuild embeddings")
            console.print("    [cyan]sqlgenx repair " + workspace_name + "[/cyan]")
    
    # Load and display schema info if available
    if schema_sql.exists() and schema_sql.stat().st_size > 0:
        console.print()
        console.print("[bold]Schema Content:[/bold]")
        
        try:
            meta = workspace_manager._load_meta(workspace_dir)
            dialect = normalize_dialect(meta.dbms_type if meta else "")
            
            loader = SchemaLoader(dialect=dialect)
            schema_info = loader.load_from_file(schema_sql)
            
            console.print(f"  Tables: [cyan]{len(schema_info.tables)}[/cyan]")
            
            if schema_info.tables:
                console.print(f"  Table names:")
                for i, table in enumerate(schema_info.tables[:10]):
                    table_name = table.name if table.name and table.name.strip() else f"unnamed_table_{i+1}"
                    console.print(f"    {i+1}. {table_name} ({len(table.columns)} columns)")
                    # Show first 2 column names as example
                    for j, column in enumerate(table.columns[:2]):  # Show first 2 columns as example
                        console.print(f"        - {column.name} ({column.data_type})")
                    if len(table.columns) > 2:
                        console.print(f"        ... and {len(table.columns) - 2} more columns")
                
                if len(schema_info.tables) > 10:
                    console.print(f"    ... and {len(schema_info.tables) - 10} more")
            else:
                print_warning("  No tables found in schema!")
        
        except Exception as e:
            print_error(f"  Failed to parse schema: {str(e)}")
    
    console.print()


def repair_workspace(workspace: Optional[str] = None, force: bool = False) -> None:
    """Repair workspace by rebuilding embeddings."""
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        raise typer.Exit(1)
    
    schema_file = workspace_dir / "schema.sql"
    if not schema_file.exists():
        print_error("Schema file not found - cannot repair")
        print_info("Try reconnecting to the database:")
        console.print(f"  [cyan]sqlgenx connect --url \"...\" --name {workspace_name}[/cyan]")
        raise typer.Exit(1)
    
    console.print()
    console.print(f"[bold yellow]🔧 Repairing workspace: {workspace_name}[/bold yellow]")
    console.print()
    
    if not force:
        if not confirm("This will rebuild embeddings. Continue?", default=True):
            print_info("Repair cancelled")
            return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Loading schema...", total=None)
        
        try:
            meta = workspace_manager._load_meta(workspace_dir)
            dialect = meta.dbms_type if meta and meta.dbms_type != "generic" else ""
            
            loader = SchemaLoader(dialect=dialect)
            schema_info = loader.load_from_file(schema_file)
            
            if not schema_info.tables:
                progress.stop()
                console.print()
                print_error("No tables found in schema")
                raise typer.Exit(1)
            
            progress.update(task, description="✓ Schema loaded")
            
            # Rebuild embeddings
            progress.update(task, description="Rebuilding embeddings...")
            
            vector_store = VectorStore(workspace_dir)
            vector_store.index_schema(schema_info)
            
            progress.update(task, description="✓ Repair complete")
        
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Repair failed: {str(e)}")
            raise typer.Exit(1)
    
    console.print()
    print_success(f"Workspace '{workspace_name}' repaired successfully!")
    print_info(f"Rebuilt embeddings for {len(schema_info.tables)} tables")