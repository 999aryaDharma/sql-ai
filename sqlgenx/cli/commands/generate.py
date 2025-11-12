"""Generate command for SQLGenX CLI."""
from typing import Optional
import typer
import sys
import os
from rich.markdown import Markdown


# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import (
    print_success, print_error, print_warning, print_info, print_sql, 
    console, print_context_info
)
from sqlgenx.core.graph import SQLGenerationGraph
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel


def generate_sql(
    query: str,
    workspace: Optional[str],
    explain: bool,
    copy: bool
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
            graph = SQLGenerationGraph(api_key)
            
            progress.update(task, description="🔍 Retrieving schema context...")
            
            result = graph.run(
                workspace_dir=workspace_dir,
                user_query=query,
                dbms_type=meta.dbms_type
            )
            
            progress.update(task, description="✓ SQL generated successfully")
        
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to generate SQL: {str(e)}")
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
    
    print_sql(final_sql, title=f"Generated SQL ({meta.dbms_type})")
    
    # Show validation warnings
    validation_result = result.get("validation_result")
    if validation_result and validation_result.warnings:
        console.print()
        print_warning("Validation warnings:")
        for warning in validation_result.warnings:
            console.print(f"  • {warning}", style="yellow")
    
    # Explain query if requested
    if explain:
        from sqlgenx.core.llm_engine import LLMEngine
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating explanation...", total=None)
            try:
                llm = LLMEngine(api_key)
                explanation = llm.explain_query_as_markdown(final_sql)
                progress.update(task, description="✓ Explanation ready")
            except Exception as e:
                progress.stop()
                print_warning(f"Failed to generate explanation: {str(e)}")
                explanation = None
        
        if explanation:
            console.print()
            panel = Panel(explanation, title="📖 Query Explanation", border_style="cyan2")
            console.print(panel)
    
    # Copy to clipboard if requested
    if copy:
        try:
            import pyperclip
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