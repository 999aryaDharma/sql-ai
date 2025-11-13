"""Database connection commands for SQLGenX."""
from typing import Optional
import typer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import (
    print_success, print_error, print_info, console, confirm
)
from sqlgenx.core.db_connector import (
    DatabaseConnection, ConnectionManager, build_connection_string
)
from sqlgenx.utils.helpers import normalize_dialect
from sqlgenx.core.schema_loader import SchemaLoader, SchemaInfo, TableInfo, ColumnInfo
from sqlgenx.core.vector_store import VectorStore
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt
import getpass


def connect_interactive() -> None:
    """Interactive database connection setup."""
    console.print("\n[bold cyan]🔌 Database Connection Setup[/bold cyan]\n")
    
    # Get connection details
    dialect = Prompt.ask(
        "Database type",
        choices=["postgresql", "mysql", "sqlite"],
        default="postgresql"
    )
    
    if dialect == "sqlite":
        database = Prompt.ask("Database file path")
        connection_string = f"sqlite:///{database}"
        host = port = username = password = None
    else:
        host = Prompt.ask("Host", default="localhost")
        
        default_port = "5432" if dialect == "postgresql" else "3306"
        port = Prompt.ask("Port", default=default_port)
        
        database = Prompt.ask("Database name")
        username = Prompt.ask("Username")
        password = getpass.getpass("Password: ")
        
        connection_string = build_connection_string(
            dialect=dialect,
            host=host,
            port=int(port),
            database=database,
            username=username,
            password=password
        )
    
    # Test connection
    console.print("\n[dim]Testing connection...[/dim]")
    try:
        db = DatabaseConnection(connection_string)
        if db.test_connection():
            print_success("Connection successful! ✓")
        else:
            print_error("Connection test failed")
            raise typer.Exit(1)
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        raise typer.Exit(1)
    
    # Get workspace details
    console.print()
    workspace_name = Prompt.ask("Workspace name", default=database)
    description = Prompt.ask("Description (optional)", default="")
    
    # Import schema
    console.print()
    if confirm("Import database schema to workspace?", default=True):
        import_schema_from_connection(
            connection_string=connection_string,
            workspace_name=workspace_name,
            description=description if description else None,
            save_connection=True
        )
    
    db.close()


def connect_from_string(
    connection_string: str,
    name: str,
    description: Optional[str] = None,
    import_schema: bool = True
) -> None:
    """Connect to database using connection string."""
    
    # Test connection
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Testing connection...", total=None)
        
        try:
            db = DatabaseConnection(connection_string)
            if not db.test_connection():
                progress.stop()
                print_error("Connection test failed")
                raise typer.Exit(1)
            
            progress.update(task, description="✓ Connection successful")
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Connection failed: {str(e)}")
            raise typer.Exit(1)
    
    console.print()
    print_success("Connected to database successfully!")
    
    # Import schema if requested
    if import_schema:
        import_schema_from_connection(
            connection_string=connection_string,
            workspace_name=name,
            description=description,
            save_connection=True
        )
    else:
        # Just save the connection
        workspace_manager = WorkspaceManager()
        workspace_dir = workspace_manager.config_manager.get_workspace_dir(name)
        
        if not workspace_dir.exists():
            print_error(f"Workspace '{name}' does not exist")
            print_info("Create it first with: sqlgenx load <schema.sql>")
            raise typer.Exit(1)
        
        conn_manager = ConnectionManager(workspace_dir)
        conn_manager.save_connection(name, connection_string, description)
        print_success(f"Connection saved to workspace '{name}'")
    
    db.close()


def import_schema_from_connection(
    connection_string: str,
    workspace_name: str,
    description: Optional[str] = None,
    save_connection: bool = True
) -> None:
    """Import schema from live database connection."""
    
    workspace_manager = WorkspaceManager()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Connect to database
        task = progress.add_task("Connecting to database...", total=None)
        try:
            db = DatabaseConnection(connection_string)
            progress.update(task, description="✓ Connected")
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to connect: {str(e)}")
            raise typer.Exit(1)
        
        # Extract schema
        progress.update(task, description="Extracting schema...")
        try:
            schema_data = db.get_schema_info()
            progress.update(task, description="✓ Schema extracted")
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to extract schema: {str(e)}")
            db.close()
            raise typer.Exit(1)
        
        # Convert to SchemaInfo format
        progress.update(task, description="Converting schema...")
        schema_info = convert_db_schema_to_schema_info(schema_data)
        
        # Create or update workspace
        progress.update(task, description="Creating workspace...")
        try:
            workspace_dir = workspace_manager.config_manager.get_workspace_dir(workspace_name)
            
            if workspace_dir.exists():
                if not confirm(f"Workspace '{workspace_name}' exists. Overwrite?", default=False):
                    progress.stop()
                    print_info("Operation cancelled")
                    db.close()
                    return
                
                # Update existing workspace
                workspace_manager.switch_workspace(workspace_name)
            else:
                # Create new workspace
                workspace_dir.mkdir(parents=True, exist_ok=True)
                (workspace_dir / "embeddings").mkdir(exist_ok=True)
                (workspace_dir / "history").mkdir(exist_ok=True)
                
                # Save metadata
                from datetime import datetime
                from sqlgenx.utils.workspace_manager import WorkspaceMeta
                
                meta = WorkspaceMeta(
                    name=workspace_name,
                    schema_path="<live_database>",
                    dbms_type=normalize_dialect(schema_data["dialect"]),
                    created_at=datetime.now().isoformat(),
                    last_used=datetime.now().isoformat(),
                    description=description
                )
                workspace_manager._save_meta(workspace_dir, meta)
            
            progress.update(task, description="✓ Workspace ready")
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to create workspace: {str(e)}")
            db.close()
            raise typer.Exit(1)
        
        # Save schema SQL
        progress.update(task, description="Saving schema...")
        schema_sql = generate_schema_sql(schema_info)
        schema_file = workspace_dir / "schema.sql"
        schema_file.write_text(schema_sql)
        
        # IMPORTANT: Also save raw_sql in schema_info for consistency
        schema_info.raw_sql = schema_sql
        
        # Create embeddings
        progress.update(task, description="Creating embeddings...")
        try:
            vector_store = VectorStore(workspace_dir)
            vector_store.index_schema(schema_info)
            progress.update(task, description="✓ Embeddings created")
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to create embeddings: {str(e)}")
            db.close()
            raise typer.Exit(1)
        
        # Save connection
        if save_connection:
            progress.update(task, description="Saving connection...")
            conn_manager = ConnectionManager(workspace_dir)
            conn_manager.save_connection(workspace_name, connection_string, description)
        
        # Switch to workspace
        progress.update(task, description="Activating workspace...")
        workspace_manager.switch_workspace(workspace_name)
        
        progress.update(task, description="✓ Import complete")
        db.close()
    
    # Success message
    console.print()
    print_success(f"Schema imported successfully into workspace '{workspace_name}'")
    print_info(f"Found {len(schema_info.tables)} tables")
    
    # Show tables
    for table in schema_info.tables[:10]:
        console.print(f"  • [cyan]{table.name}[/cyan]")
    
    if len(schema_info.tables) > 10:
        console.print(f"  [dim]... and {len(schema_info.tables) - 10} more[/dim]")
    
    console.print()
    print_info("Generate queries with:")
    console.print(f"  [cyan]sqlgenx generate \"your query here\"[/cyan]")
    
    if save_connection:
        console.print()
        print_info("Execute queries with:")
        console.print(f"  [cyan]sqlgenx generate \"query\" --run[/cyan]")


def list_connections(workspace: Optional[str] = None) -> None:
    """List saved database connections."""
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        raise typer.Exit(1)
    
    conn_manager = ConnectionManager(workspace_dir)
    connections = conn_manager.list_connections()
    
    if not connections:
        console.print()
        print_info(f"No saved connections in workspace '{workspace_name}'")
        console.print()
        print_info("Connect to a database with:")
        console.print("  [cyan]sqlgenx connect[/cyan]")
        return
    
    console.print()
    table = Table(title=f"Connections in '{workspace_name}'", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Dialect", style="yellow")
    table.add_column("Description", style="dim")
    
    for conn in connections:
        table.add_row(
            conn["name"],
            conn.get("dialect", "unknown"),
            conn.get("description", "-")
        )
    
    console.print(table)


def convert_db_schema_to_schema_info(schema_data: dict) -> SchemaInfo:
    """Convert database schema data to SchemaInfo object."""
    tables = []
    
    for table_data in schema_data["tables"]:
        # Ensure table name is not empty - get raw name and ensure it's a string
        raw_name = table_data.get("name", "")
        # Make sure it's a string, not a tuple or other type
        table_name = str(raw_name) if raw_name is not None else ""
        
        if not table_name or not table_name.strip():
            table_name = f"unnamed_table_{len(tables)+1}"
        
        columns = []
        for col_data in table_data["columns"]:
            col = ColumnInfo(
                name=col_data["name"],
                data_type=col_data["type"],
                nullable=col_data.get("nullable", True),
                primary_key=col_data.get("primary_key", False),
                foreign_key=col_data["name"] in table_data.get("foreign_keys", {}),
                default=col_data.get("default")
            )
            columns.append(col)
        
        table = TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=table_data.get("primary_keys", []),
            foreign_keys=table_data.get("foreign_keys", {}),
            indexes=table_data.get("indexes", []),
            column_samples=table_data.get("column_samples", [])
        )
        tables.append(table)
    
    # Generate SQL representation
    raw_sql = generate_schema_sql(SchemaInfo(tables=tables, raw_sql=""))
    
    return SchemaInfo(tables=tables, raw_sql=raw_sql)


def generate_schema_sql(schema_info: SchemaInfo) -> str:
    """Generate SQL DDL from SchemaInfo."""
    sql_parts = []
    
    for table in schema_info.tables:
        table_sql = [f"CREATE TABLE {table.name} ("]
        
        col_defs = []
        for col in table.columns:
            col_def = f"    {col.name} {col.data_type}"
            
            if col.primary_key:
                col_def += " PRIMARY KEY"
            
            if not col.nullable:
                col_def += " NOT NULL"
            
            if col.default:
                col_def += f" DEFAULT {col.default}"
            
            col_defs.append(col_def)
        
        # Add foreign keys
        for col_name, ref in table.foreign_keys.items():
            col_defs.append(f"    FOREIGN KEY ({col_name}) REFERENCES {ref}")
        
        table_sql.append(",\n".join(col_defs))
        table_sql.append(");")
        
        sql_parts.append("\n".join(table_sql))
    
    return "\n\n".join(sql_parts)