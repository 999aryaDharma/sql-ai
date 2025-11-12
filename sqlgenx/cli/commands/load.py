"""Load command for SQLGenX CLI."""
from pathlib import Path
from typing import Optional
import typer
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager
from sqlgenx.utils.rich_helpers import print_success, print_error, print_info, console
from sqlgenx.core.schema_loader import SchemaLoader
from sqlgenx.core.vector_store import VectorStore
from rich.progress import Progress, SpinnerColumn, TextColumn


def load_schema(
    schema_file: Path, 
    name: Optional[str],
    dbms: str,
    description: Optional[str]
) -> None:
    """Load a SQL schema file into a workspace."""
    
    # Validate schema file exists
    if not schema_file.exists():
        print_error(f"Schema file not found: {schema_file}")
        raise typer.Exit(1)
    
    # Validate file is readable
    if not schema_file.is_file():
        print_error(f"Path is not a file: {schema_file}")
        raise typer.Exit(1)
    
    # Generate workspace name if not provided
    if not name:
        name = schema_file.stem
    
    # Validate workspace name
    if not name or not name.strip():
        print_error("Workspace name cannot be empty")
        raise typer.Exit(1)
    
    workspace_manager = WorkspaceManager()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Create workspace
        task = progress.add_task("Creating workspace...", total=None)
        try:
            workspace_dir = workspace_manager.create_workspace(
                name=name,
                schema_path=str(schema_file),
                dbms_type=dbms,
                description=description
            )
            progress.update(task, description="✓ Workspace created")
        except ValueError as e:
            progress.stop()
            print_error(str(e))
            raise typer.Exit(1)
        except Exception as e:
            progress.stop()
            print_error(f"Failed to create workspace: {str(e)}")
            raise typer.Exit(1)
        
        # Load and parse schema
        progress.update(task, description="Parsing schema...")
        try:
            loader = SchemaLoader(dialect=dbms if dbms != "generic" else "")
            schema_info = loader.load_from_file(schema_file)
            progress.update(task, description="✓ Schema parsed")
            
            if not schema_info.tables:
                progress.stop()
                print_error("No tables found in schema file")
                print_info("Make sure your schema file contains valid CREATE TABLE statements")
                raise typer.Exit(1)
            
        except Exception as e:
            progress.stop()
            print_error(f"Failed to parse schema: {str(e)}")
            print_info("Check if your SQL file is valid and properly formatted")
            raise typer.Exit(1)
        
        # Create embeddings
        progress.update(task, description="Creating embeddings...")
        try:
            vector_store = VectorStore(workspace_dir)
            vector_store.index_schema(schema_info)
            progress.update(task, description="✓ Embeddings created")
        except Exception as e:
            progress.stop()
            print_error(f"Failed to create embeddings: {str(e)}")
            raise typer.Exit(1)
        
        # Switch to new workspace
        progress.update(task, description="Activating workspace...")
        try:
            workspace_manager.switch_workspace(name)
            progress.update(task, description="✓ Workspace activated")
        except Exception as e:
            progress.stop()
            print_error(f"Failed to activate workspace: {str(e)}")
            raise typer.Exit(1)
    
    # Success message
    console.print()
    print_success(f"Schema loaded successfully into workspace '{name}'")
    print_info(f"Found {len(schema_info.tables)} tables:")
    
    # Show table names
    table_names = [t.name for t in schema_info.tables]
    if len(table_names) <= 10:
        for table in table_names:
            console.print(f"  • [cyan]{table}[/cyan]")
    else:
        for table in table_names[:10]:
            console.print(f"  • [cyan]{table}[/cyan]")
        console.print(f"  [dim]... and {len(table_names) - 10} more[/dim]")
    
    console.print()
    print_info(f"Workspace '{name}' is now active. Try generating a query:")
    console.print(f"  [bold cyan]sqlgenx generate \"your query here\"[/bold cyan]")