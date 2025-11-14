"""
sqlgenx/cli/commands/validate.py (NEW FILE)

Standalone validation command for debugging.
"""

from typing import Optional
import typer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import print_success, print_error, print_info, console
from sqlgenx.core.semantic_validator import DeterministicSemanticValidator
from rich.table import Table
from rich.panel import Panel


def validate_query(
    query: str,
    workspace: Optional[str] = None,
    min_relevance: float = 0.2,
    n_results: int = 10
) -> None:
    """
    Validate a natural language query and show detailed debug info.
    
    This command shows exactly what happens during semantic validation:
    - Query intent extraction
    - Semantic search results
    - Relevance scores
    - Capability detection
    - Validation decision
    """
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        raise typer.Exit(1)
    
    console.print()
    console.print(f"[bold cyan]🔍 Validating Query in Workspace: {workspace_name}[/bold cyan]")
    console.print()
    console.print(f"Query: [yellow]{query}[/yellow]")
    console.print()
    
    # Create validator with debug enabled
    validator = DeterministicSemanticValidator(workspace_dir, debug=True)
    
    # Run validation
    is_valid, reason, contexts = validator.validate(
        user_query=query,
        n_results=n_results,
        min_relevance=min_relevance
    )
    
    # Show result
    console.print()
    console.print("=" * 60)
    console.print()
    
    if is_valid:
        print_success(f"✅ Query is VALID")
        console.print()
        console.print(f"Reason: {reason}")
        console.print()
        
        if contexts:
            console.print(f"[bold]Relevant Tables Found:[/bold] {len(contexts)}")
            
            table = Table(show_header=True, header_style="bold violet")
            table.add_column("Table", style="cyan")
            table.add_column("Relevance", style="yellow")
            table.add_column("Type", style="dim")
            
            for ctx in contexts:
                table_name = ctx.get('table', 'unknown')
                score = ctx.get('relevance_score', 0)
                ctx_type = ctx.get('type', 'unknown')
                
                table.add_row(
                    table_name,
                    f"{score:.3f}",
                    ctx_type
                )
            
            console.print(table)
    else:
        print_error(f"❌ Query is INVALID")
        console.print()
        console.print(f"Reason: {reason}")
        console.print()
        
        print_info("Suggestions:")
        console.print("  1. Lower relevance threshold: --min-relevance 0.1")
        console.print("  2. Increase results: --n-results 15")
        console.print("  3. Enrich semantic profile: sqlgenx enrich --force")
        console.print("  4. Check if semantic profile exists")
    
    console.print()