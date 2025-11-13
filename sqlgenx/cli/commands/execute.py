"""Execute SQL queries and preview data."""
from typing import Optional
import typer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import print_success, print_error, print_info, console
from sqlgenx.core.db_connector import DatabaseConnection, ConnectionManager
from sqlgenx.core.data_analyzer import DataAnalyzerEnhanced as DataAnalyzer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
import pandas as pd


def execute_query(
    query: str,
    workspace: Optional[str] = None,
    limit: Optional[int] = None,
    analyze: bool = False,
    export: Optional[str] = None
) -> None:
    """Execute a SQL query directly."""
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        raise typer.Exit(1)
    
    # Get database connection
    conn_manager = ConnectionManager(workspace_dir)
    connections = conn_manager.list_connections()
    
    if not connections:
        console.print()
        print_error("No database connection found")
        print_info("Connect to a database first:")
        console.print("  [cyan]sqlgenx connect[/cyan]")
        raise typer.Exit(1)
    
    conn_info = connections[0]
    connection_string = conn_info.get("connection_string")
    
    # Execute query
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("⚡ Executing query...", total=None)
        
        try:
            db = DatabaseConnection(connection_string)
            df, metadata = db.execute_query(query, limit=limit)
            progress.update(task, description="✓ Query executed")
            db.close()
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Query execution failed: {str(e)}")
            raise typer.Exit(1)
    
    # Display results
    console.print()
    print_success(f"Retrieved {len(df)} rows")
    
    display_results_table(df, title="Query Results")
    
    # Export if requested
    if export:
        try:
            if export.endswith('.csv'):
                df.to_csv(export, index=False)
            elif export.endswith('.json'):
                df.to_json(export, orient='records', indent=2)
            elif export.endswith('.xlsx'):
                df.to_excel(export, index=False)
            else:
                print_error(f"Unsupported export format: {export}")
                raise typer.Exit(1)
            
            console.print()
            print_success(f"Results exported to {export}")
        except Exception as e:
            console.print()
            print_error(f"Export failed: {str(e)}")
    
    # Analyze if requested
    if analyze:
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🧠 Analyzing results...", total=None)
            
            api_key = config_manager.gemini_api_key
            if not api_key:
                progress.stop()
                print_error("API key not configured for analysis")
                raise typer.Exit(1)
            
            analyzer = DataAnalyzer(api_key)
            result = analyzer.analyze_results(
                df=df,
                original_query="Direct SQL execution",
                sql_query=query,
                metadata=metadata
            )
            
            # The function returns a tuple of (analysis_text, metadata_dict)
            if isinstance(result, tuple):
                insights, analysis_metadata = result
            else:
                insights = result
            
            progress.update(task, description="✓ Analysis complete")
        
        console.print()
        # Use Markdown rendering for better formatting
        from rich.markdown import Markdown
        md = Markdown(insights)
        panel = Panel(
            md,
            title="🎯 AI Insights & Analysis",
            border_style="green",
            padding=(1, 2)
        )
        console.print(panel)


def preview_table(
    table_name: str,
    workspace: Optional[str] = None,
    limit: int = 10,
    stats: bool = False
) -> None:
    """Preview data from a table."""
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        raise typer.Exit(1)
    
    # Get database connection
    conn_manager = ConnectionManager(workspace_dir)
    connections = conn_manager.list_connections()
    
    if not connections:
        console.print()
        print_error("No database connection found")
        print_info("Connect to a database first:")
        console.print("  [cyan]sqlgenx connect[/cyan]")
        raise typer.Exit(1)
    
    conn_info = connections[0]
    connection_string = conn_info.get("connection_string")
    
    # Get preview
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Loading preview of {table_name}...", total=None)
        
        try:
            db = DatabaseConnection(connection_string)
            df = db.get_table_preview(table_name, limit=limit)
            
            if stats:
                table_stats = db.get_table_stats(table_name)
            
            progress.update(task, description="✓ Preview loaded")
            db.close()
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to load preview: {str(e)}")
            raise typer.Exit(1)
    
    # Display preview
    console.print()
    display_results_table(df, title=f"Preview: {table_name} (showing {len(df)} rows)")
    
    # Display stats if requested
    if stats:
        console.print()
        console.print(f"[bold]Table Statistics:[/bold]")
        console.print(f"  Total Rows: [cyan]{table_stats['row_count']:,}[/cyan]")
        console.print(f"  Columns: [cyan]{table_stats['column_count']}[/cyan]")
        console.print(f"  Column Names: {', '.join(table_stats['columns'])}")


def display_results_table(df: pd.DataFrame, title: str = "Results") -> None:
    """Display DataFrame as a rich table."""
    
    if df.empty:
        print_info("No results to display")
        return
    
    table = Table(
        title=title,
        show_header=True,
        header_style="bold violet",
        show_lines=False
    )
    
    # Add columns
    for col in df.columns:
        table.add_column(str(col), style="cyan")
    
    # Add rows (limit display)
    display_limit = min(50, len(df))
    for idx in range(display_limit):
        row = df.iloc[idx]
        table.add_row(*[str(val) if pd.notna(val) else "[dim]NULL[/dim]" for val in row])
    
    console.print(table)
    
    if len(df) > display_limit:
        console.print(f"\n[dim]... showing {display_limit} of {len(df)} rows[/dim]")


def compare_queries(
    query1: str,
    query2: str,
    workspace: Optional[str] = None,
    context: str = ""
) -> None:
    """Compare results of two queries."""
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        raise typer.Exit(1)
    
    # Get database connection
    conn_manager = ConnectionManager(workspace_dir)
    connections = conn_manager.list_connections()
    
    if not connections:
        console.print()
        print_error("No database connection found")
        raise typer.Exit(1)
    
    conn_info = connections[0]
    connection_string = conn_info.get("connection_string")
    
    # Execute both queries
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Executing queries...", total=None)
        
        try:
            db = DatabaseConnection(connection_string)
            
            progress.update(task, description="Executing query 1...")
            df1, _ = db.execute_query(query1)
            
            progress.update(task, description="Executing query 2...")
            df2, _ = db.execute_query(query2)
            
            progress.update(task, description="✓ Queries executed")
            db.close()
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Query execution failed: {str(e)}")
            raise typer.Exit(1)
    
    # Display results
    console.print()
    console.print("[bold]Query 1 Results:[/bold]")
    display_results_table(df1, title="Query 1")
    
    console.print()
    console.print("[bold]Query 2 Results:[/bold]")
    display_results_table(df2, title="Query 2")
    
    # Compare with AI
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("🧠 Comparing results...", total=None)
        
        api_key = config_manager.gemini_api_key
        if not api_key:
            progress.stop()
            print_error("API key not configured")
            raise typer.Exit(1)
        
        analyzer = DataAnalyzer(api_key)
        comparison = analyzer.compare_results(df1, df2, context)
        
        progress.update(task, description="✓ Comparison complete")
    
    console.print()
    panel = Panel(
        comparison,
        title="📊 Comparison Analysis",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)