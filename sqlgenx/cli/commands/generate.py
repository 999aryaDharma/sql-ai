"""Generate command for SQLGenX CLI."""
from typing import Optional
import typer
import sys
import os
from sqlgenx.core.data_analyzer import DataAnalyzer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table as RichTable
from sqlgenx.core.db_connector import ConnectionManager, DatabaseConnection
import pyperclip
from sqlgenx.core.llm_engine import LLMEngine
import traceback


# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import (
    print_success, print_error, print_warning, print_info, print_sql, 
    console, print_context_info
)
from sqlgenx.utils.helpers import normalize_dialect
from sqlgenx.core.graph import SQLGenerationGraph
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel


def generate_sql(
    query: str,
    workspace: Optional[str],
    explain: bool,
    copy: bool,
    run: bool = False,
    limit: Optional[int] = None,
    analyze: bool = False,
    fast: bool = False
) -> None:
    """Generate SQL from natural language query."""
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    # Determine which workspace to use
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        print_info("Available commands:")
        console.print("  • [cyan]sqlgenx list[/cyan] - List all workspaces")
        console.print("  • [cyan]sqlgenx load <schema.sql>[/cyan] - Create a new workspace")
        raise typer.Exit(1)
    
    # Get workspace metadata
    meta = workspace_manager._load_meta(workspace_dir)
    if not meta:
        print_error("Workspace metadata not found")
        print_info(f"Try recreating the workspace with: sqlgenx load <schema.sql> --name {workspace_name}")
        raise typer.Exit(1)
    
    # Check for API key
    api_key = config_manager.gemini_api_key
    if not api_key:
        print_error("Gemini API key not configured")
        console.print()
        print_info("Configure your API key in one of these ways:")
        console.print("  1. Environment variable:")
        console.print("     [cyan]export GEMINI_API_KEY='your-key'[/cyan]")
        console.print()
        console.print("  2. Create a .env file in your project:")
        console.print("     [cyan]GEMINI_API_KEY=your-key[/cyan]")
        console.print()
        console.print("  Get your API key from: [link]https://makersuite.google.com/app/apikey[/link]")
        raise typer.Exit(1)
    
    # Generate SQL
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("🤖 Analyzing query...", total=None)
        
        try:
            # Check if schema file exists
            schema_file = workspace_dir / "schema.sql"
            if not schema_file.exists():
                progress.stop()
                console.print()
                print_error("Schema file not found in workspace")
                print_info("This workspace may need to be resynced")
                print_info("Try: sqlgenx sync or reconnect with: sqlgenx connect")
                raise typer.Exit(1)
            
            graph = SQLGenerationGraph(api_key, workspace_dir=workspace_dir)

            if fast:
                console.print("[dim]⚡ Fast mode enabled: skipping optimization step[/dim]")
            
            progress.update(task, description="🔍 Retrieving schema context...")
            
            result = graph.run(
                workspace_dir=workspace_dir,
                user_query=query,
                dbms_type=normalize_dialect(meta.dbms_type)
            )
            
            progress.update(task, description="✓ SQL generated successfully")
        
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to generate SQL: {str(e)}")
            print_info(f"Error details: {traceback.format_exc()}")
            print_info("Check your API key and internet connection")
            raise typer.Exit(1)
    
    # Check for errors
    if result.get("errors"):
        console.print()
        print_error("Generation failed with errors:")
        for error in result["errors"]:
            console.print(f"  • {error}", style="red")
        raise typer.Exit(1)
    
    # Display results
    console.print()
    
    # Show retrieved context
    if result.get("retrieved_contexts"):
        print_context_info(result["retrieved_contexts"])
        console.print()
    
    # Display SQL
    final_sql = result.get("final_sql", "")
    if not final_sql:
        print_error("No SQL was generated")
        raise typer.Exit(1)
    
    print_sql(final_sql, title=f"Generated SQL ({meta.dbms_type})", line_numbers=False)
    
    # Show validation warnings
    validation_result = result.get("validation_result")
    if validation_result and validation_result.warnings:
        console.print()
        print_warning("Validation warnings:")
        for warning in validation_result.warnings:
            console.print(f"  • {warning}", style="yellow")
    
    # Explain query if requested
    if explain:
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating explanation...", total=None)
            try:
                llm = LLMEngine(api_key)
                explanation = llm.explain_query(final_sql)
                progress.update(task, description="✓ Explanation ready")
            except Exception as e:
                progress.stop()
                print_warning(f"Failed to generate explanation: {str(e)}")
                explanation = None
        
        if explanation:
            console.print()
            panel = Panel(explanation, title="📖 Query Explanation", border_style="blue")
            console.print(panel)
    
    # Copy to clipboard if requested
    if copy:
        try:
            pyperclip.copy(final_sql)
            console.print()
            print_success("SQL copied to clipboard! ✂️")
        except ImportError:
            console.print()
            print_warning("Clipboard feature requires pyperclip")
            print_info("Install with: pip install pyperclip")
        except Exception as e:
            console.print()
            print_warning(f"Failed to copy to clipboard: {str(e)}")
    
    # Execute query if requested
    if run:
        console.print()
        console.print("[bold yellow]⚡ Executing query...[/bold yellow]")
        
        # Check if connection exists
        conn_manager = ConnectionManager(workspace_dir)
        connections = conn_manager.list_connections()
        
        if not connections:
            console.print()
            print_error("No database connection found in workspace")
            print_info("Connect to a database first:")
            console.print("  [cyan]sqlgenx connect[/cyan]")
            raise typer.Exit(1)
        
        # Use first connection (or could ask user to choose)
        conn_info = connections[0]
        connection_string = conn_info.get("connection_string")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing query...", total=None)
                
                db = DatabaseConnection(connection_string)
                df, exec_metadata = db.execute_query(final_sql, limit=limit)
                
                progress.update(task, description="✓ Query executed")
                db.close()
            
            console.print()
            print_success(f"Query executed successfully! Retrieved {len(df)} rows")
            
            # Display results
            console.print()
            results_table = RichTable(
                title="Query Results",
                show_header=True,
                header_style="bold violet"
            )
            
            # Add columns
            for col in df.columns:
                results_table.add_column(str(col), style="cyan")
            
            # Add rows (limit display to 50 rows)
            display_limit = min(50, len(df))
            for idx in range(display_limit):
                row = df.iloc[idx]
                results_table.add_row(*[str(val) for val in row])
            
            console.print(results_table)
            
            if len(df) > display_limit:
                console.print(f"\n[dim]... showing {display_limit} of {len(df)} rows[/dim]")
            
            # Analyze results if requested
            if analyze:
                console.print()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("🧠 Analyzing results with AI...", total=None)
                    

                    analyzer = DataAnalyzer(api_key)
                    result = analyzer.analyze_results(
                        df=df,
                        original_query=query,
                        sql_query=final_sql,
                        metadata=exec_metadata
                    )
                    
                    # The function returns a tuple of (analysis_text, metadata_dict)
                    if isinstance(result, tuple):
                        insights, analysis_metadata = result
                    else:
                        insights = result
                    
                    progress.update(task, description="✓ Analysis complete")
                
                console.print()
                md = Markdown(insights)
                panel = Panel(
                    md,
                    title="🎯 AI Insights & Analysis",
                    border_style="cyan2",
                    padding=(1, 2)
                )
                console.print(panel)
        
            # At the end of generate_sql function in generate.py
            from sqlgenx.core.rate_limiting.limiter import global_rate_limiter

            stats = global_rate_limiter.get_current_usage()
            if stats['rate_limit_hits'] > 0:
                console.print(f"\n[dim]Rate limit hits: {stats['rate_limit_hits']}[/dim]")
                console.print(f"[dim]Total wait time: {stats['total_wait_time']:.1f}s[/dim]")
        
        except Exception as e:
            console.print()
            print_error(f"Failed to execute query: {str(e)}")
            print_info("Make sure the connection is valid and the query is correct")