"""
sqlgenx/cli/commands/enrich.py

CLI command for semantic enrichment.
"""
from typing import Optional
import typer
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlgenx.utils.workspace_manager import WorkspaceManager, ConfigManager
from sqlgenx.utils.rich_helpers import (
    print_success, print_error, print_info, print_warning, console, confirm
)
from sqlgenx.core.schema_loader import SchemaLoader
from sqlgenx.core.semantic_enricher import SemanticEnrichmentPipeline
from sqlgenx.core.vector_store import VectorStore
from sqlgenx.utils.helpers import normalize_dialect
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import json


def enrich_workspace(
    workspace: Optional[str] = None,
    force: bool = False,
    skip_llm: bool = False
) -> None:
    """
    Run semantic enrichment on workspace schema.
    
    Args:
        workspace: Workspace name (uses current if not specified)
        force: Force re-enrichment even if profile exists
        skip_llm: Skip LLM enrichment (static only)
    """
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    # Determine workspace
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        console.print()
        print_info("See available workspaces:")
        console.print("  [cyan]sqlgenx list[/cyan]")
        raise typer.Exit(1)
    
    # Check if schema exists
    schema_file = workspace_dir / "schema.sql"
    if not schema_file.exists():
        print_error("Schema file not found in workspace")
        print_info("Load a schema first:")
        console.print("  [cyan]sqlgenx load <schema.sql>[/cyan]")
        raise typer.Exit(1)
    
    # Check if already enriched
    semantic_profile = workspace_dir / "semantic_profile.json"
    if semantic_profile.exists() and not force:
        console.print()
        print_warning("Semantic profile already exists")
        
        if not confirm("Re-run enrichment?", default=False):
            print_info("Enrichment cancelled")
            return
    
    # Get API key if not skipping LLM
    if not skip_llm:
        api_key = config_manager.gemini_api_key
        if not api_key:
            print_error("Gemini API key not configured")
            console.print()
            print_info("Set API key with:")
            console.print("  [cyan]export GEMINI_API_KEY='your-key'[/cyan]")
            console.print()
            print_warning("Or use --skip-llm for rule-based enrichment only")
            raise typer.Exit(1)
    else:
        api_key = None
        print_info("Running in static-only mode (no LLM)")
    
    # Load metadata
    meta = workspace_manager._load_meta(workspace_dir)
    if not meta:
        print_error("Workspace metadata not found")
        raise typer.Exit(1)
    
    console.print()
    console.print(f"[bold cyan]🚀 Starting semantic enrichment for '{workspace_name}'[/bold cyan]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Load schema
        task = progress.add_task("Loading schema...", total=None)
        
        try:
            dialect = normalize_dialect(meta.dbms_type)
            loader = SchemaLoader(dialect=dialect if dialect != "generic" else "")
            schema_info = loader.load_from_file(schema_file)
            
            if not schema_info.tables:
                progress.stop()
                console.print()
                print_error("No tables found in schema")
                raise typer.Exit(1)
            
            progress.update(task, description=f"✓ Loaded {len(schema_info.tables)} tables")
            
        except Exception as e:
            progress.stop()
            console.print()
            print_error(f"Failed to load schema: {str(e)}")
            raise typer.Exit(1)
    
    # Run enrichment pipeline
    try:
        if skip_llm:
            # Static enrichment only
            from sqlgenx.core.semantic_enricher import StaticSemanticEnricher
            
            console.print()
            console.print("📊 Running static semantic enrichment...")
            
            enricher = StaticSemanticEnricher(schema_info)
            static_enrichment = enricher.enrich()
            
            # Save basic profile
            basic_profile = {
                'schema_name': workspace_name,
                'version': '1.0-static',
                'created_at': __import__('datetime').datetime.now().isoformat(),
                'static_enrichment': static_enrichment,
                'enrichment_type': 'static_only'
            }
            
            with open(semantic_profile, 'w', encoding='utf-8') as f:
                json.dump(basic_profile, f, indent=2, ensure_ascii=False)
            
            console.print()
            print_success("Static enrichment complete!")
            
        else:
            # Full pipeline with LLM
            console.print()
            pipeline = SemanticEnrichmentPipeline(
                schema_info=schema_info,
                api_key=api_key,
                workspace_dir=workspace_dir
            )
            
            semantic_schema = pipeline.run()
            
            console.print()
            print_success("Semantic enrichment complete!")
        
        # Re-index embeddings with semantic documents
        console.print()
        console.print("🔄 Re-indexing embeddings with semantic documents...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Rebuilding embeddings...", total=None)
            
            try:
                vector_store = VectorStore(workspace_dir)
                vector_store.index_schema(schema_info, use_semantic=True)
                
                progress.update(task, description="✓ Embeddings rebuilt")
                
            except Exception as e:
                progress.stop()
                console.print()
                print_warning(f"Failed to rebuild embeddings: {str(e)}")
                print_info("You can rebuild manually with: sqlgenx repair")
        
        # Display summary
        console.print()
        _display_enrichment_summary(workspace_dir, skip_llm)
        
    except Exception as e:
        console.print()
        print_error(f"Enrichment failed: {str(e)}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


def show_semantic_info(workspace: Optional[str] = None) -> None:
    """Show semantic enrichment information for workspace."""
    
    workspace_manager = WorkspaceManager()
    config_manager = ConfigManager()
    
    workspace_name = workspace or config_manager.get_current_workspace()
    workspace_dir = workspace_manager.get_workspace(workspace_name)
    
    if not workspace_dir:
        print_error(f"Workspace '{workspace_name}' not found")
        raise typer.Exit(1)
    
    semantic_profile = workspace_dir / "semantic_profile.json"
    
    if not semantic_profile.exists():
        console.print()
        print_warning(f"No semantic enrichment found for '{workspace_name}'")
        console.print()
        print_info("Run semantic enrichment with:")
        console.print("  [cyan]sqlgenx enrich[/cyan]")
        return
    
    # Load profile
    with open(semantic_profile, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    
    console.print()
    console.print(f"[bold cyan]📊 Semantic Profile: {workspace_name}[/bold cyan]")
    console.print()
    
    # Basic info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Key", style="cyan")
    info_table.add_column("Value", style="yellow")
    
    info_table.add_row("Version", profile.get('version', 'unknown'))
    info_table.add_row("Created", profile.get('created_at', 'unknown').split('T')[0])
    info_table.add_row("Type", profile.get('enrichment_type', 'full'))
    
    console.print(info_table)
    console.print()
    
    # Domain groupings
    domain_groupings = profile.get('domain_groupings', {})
    if domain_groupings:
        console.print("[bold]Domain Groupings:[/bold]")
        for domain, tables in domain_groupings.items():
            console.print(f"  • [cyan]{domain}[/cyan]: {len(tables)} tables")
        console.print()
    
    # Table types
    table_semantics = profile.get('table_semantics', [])
    if table_semantics:
        type_counts = {}
        for table in table_semantics:
            table_type = table.get('table_type', 'unknown')
            type_counts[table_type] = type_counts.get(table_type, 0) + 1
        
        console.print("[bold]Table Types:[/bold]")
        for table_type, count in type_counts.items():
            console.print(f"  • {table_type}: {count}")
        console.print()
    
    # Sample enrichment
    if table_semantics:
        sample = table_semantics[0]
        console.print(f"[bold]Sample Table Enrichment:[/bold] [cyan]{sample['table_name']}[/cyan]")
        console.print(f"  Purpose: {sample.get('business_purpose', 'N/A')}")
        console.print(f"  Domain: {sample.get('business_domain', 'N/A')}")
        console.print(f"  Type: {sample.get('table_type', 'N/A')}")
        
        derived = sample.get('derived_fields', [])
        if derived:
            console.print(f"  Metrics: {len(derived)} available")
        console.print()
    
    # Confidence notes
    confidence_notes = profile.get('confidence_notes', [])
    if confidence_notes:
        console.print("[bold yellow]⚠️  Notes:[/bold yellow]")
        for note in confidence_notes[:5]:
            console.print(f"  • {note}")
        if len(confidence_notes) > 5:
            console.print(f"  [dim]... and {len(confidence_notes) - 5} more[/dim]")


def _display_enrichment_summary(workspace_dir: Path, skip_llm: bool) -> None:
    """Display enrichment summary."""
    
    semantic_profile = workspace_dir / "semantic_profile.json"
    
    if not semantic_profile.exists():
        return
    
    with open(semantic_profile, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    
    # Build summary
    lines = []
    
    if skip_llm:
        lines.append("[bold]Static Enrichment Summary:[/bold]")
    else:
        lines.append("[bold]Semantic Enrichment Summary:[/bold]")
    
    lines.append("")
    
    # Table count
    table_semantics = profile.get('table_semantics', [])
    lines.append(f"Tables enriched: [cyan]{len(table_semantics)}[/cyan]")
    
    # Domain count
    domain_groupings = profile.get('domain_groupings', {})
    lines.append(f"Domains detected: [cyan]{len(domain_groupings)}[/cyan]")
    
    # Join paths
    join_paths = profile.get('join_paths', [])
    lines.append(f"Join paths: [cyan]{len(join_paths)}[/cyan]")
    
    # Derived fields
    total_derived = sum(
        len(t.get('derived_fields', [])) 
        for t in table_semantics
    )
    lines.append(f"Derived metrics: [cyan]{total_derived}[/cyan]")
    
    lines.append("")
    lines.append("[dim]Benefits:[/dim]")
    lines.append("  ✓ Improved query understanding")
    lines.append("  ✓ Better table retrieval")
    lines.append("  ✓ Smarter SQL generation")
    
    panel = Panel(
        "\n".join(lines),
        title="✨ Enrichment Complete",
        border_style="green"
    )
    
    console.print(panel)