"""Main CLI entry point for SQLGenX."""
import typer
from pathlib import Path
from typing import Optional
from sqlgenx.utils.rich_helpers import print_banner

app = typer.Typer(
    name="sqlgenx",
    help="AI-powered CLI for Smart SQL Generation",
    add_completion=False
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """SQLGenX - AI-powered SQL generation from natural language."""
    if ctx.invoked_subcommand is None:
        print_banner()
        typer.echo("\nUse --help to see available commands")


@app.command()
def load(
    schema_file: Path = typer.Argument(..., help="Path to SQL schema file"),
    name: str = typer.Option(None, "--name", "-n", help="Workspace name"),
    dbms: str = typer.Option("generic", "--dbms", "-d", help="Database type (mysql, postgresql, sqlite, generic)"),
    description: str = typer.Option(None, "--desc", help="Workspace description")
) -> None:
    """Load a SQL schema file into a workspace."""
    from sqlgenx.cli.commands.load import load_schema
    load_schema(schema_file, name, dbms, description)


@app.command()
def generate(
    query: str = typer.Argument(..., help="Natural language query"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to use"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Explain the generated query"),
    copy: bool = typer.Option(False, "--copy", "-c", help="Copy SQL to clipboard"),
    run: bool = typer.Option(False, "--run", "-r", help="Execute query on connected database"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of results"),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Analyze results with AI (requires --run)")
) -> None:
    """Generate SQL from natural language query."""
    from sqlgenx.cli.commands.generate import generate_sql
    generate_sql(query, workspace, explain, copy, run, limit, analyze)


@app.command()
def use(
    workspace: str = typer.Argument(..., help="Workspace name to switch to")
) -> None:
    """Switch to a different workspace."""
    from sqlgenx.cli.commands.switch import switch_workspace
    switch_workspace(workspace)


@app.command(name="list")
def list_workspaces() -> None:
    """List all available workspaces."""
    from sqlgenx.cli.commands.switch import list_all_workspaces
    list_all_workspaces()


@app.command()
def delete(
    workspace: str = typer.Argument(..., help="Workspace name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
) -> None:
    """Delete a workspace."""
    from sqlgenx.cli.commands.switch import delete_workspace_cmd
    delete_workspace_cmd(workspace, force)


@app.command()
def config(
    key: str = typer.Argument(None, help="Config key to set"),
    value: str = typer.Argument(None, help="Config value"),
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration")
) -> None:
    """Configure SQLGenX settings."""
    from sqlgenx.cli.commands.config import config_cmd
    config_cmd(key, value, show)


@app.command()
def info(
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to show info for")
) -> None:
    """Show information about current or specified workspace."""
    from sqlgenx.cli.commands.config import info_cmd
    info_cmd(workspace)


@app.command()
def connect(
    connection_string: str = typer.Option(None, "--url", help="Database connection URL"),
    name: str = typer.Option(None, "--name", "-n", help="Workspace name"),
    description: str = typer.Option(None, "--desc", help="Connection description"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive setup")
) -> None:
    """Connect to a live database."""
    from sqlgenx.cli.commands.connect import connect_interactive, connect_from_string
    
    if interactive or not connection_string:
        connect_interactive()
    else:
        if not name:
            typer.echo("Error: --name is required when using --url")
            raise typer.Exit(1)
        connect_from_string(connection_string, name, description)


@app.command()
def connections(
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to list connections for")
) -> None:
    """List saved database connections."""
    from sqlgenx.cli.commands.connect import list_connections
    list_connections(workspace)


@app.command()
def execute(
    query: str = typer.Argument(..., help="SQL query to execute"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to use"),
    limit: int = typer.Option(None, "--limit", "-l", help="Limit number of results"),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Analyze results with AI"),
    export: str = typer.Option(None, "--export", "-e", help="Export results to file (csv, json, xlsx)")
) -> None:
    """Execute a SQL query directly on the connected database."""
    from sqlgenx.cli.commands.execute import execute_query
    execute_query(query, workspace, limit, analyze, export)


@app.command()
def preview(
    table: str = typer.Argument(..., help="Table name to preview"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to use"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of rows to show"),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show table statistics")
) -> None:
    """Preview data from a table."""
    from sqlgenx.cli.commands.execute import preview_table
    preview_table(table, workspace, limit, stats)


@app.command()
def compare(
    query1: str = typer.Argument(..., help="First SQL query"),
    query2: str = typer.Argument(..., help="Second SQL query"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to use"),
    context: str = typer.Option("", "--context", "-c", help="Context for comparison")
) -> None:
    """Compare results of two SQL queries."""
    from sqlgenx.cli.commands.execute import compare_queries
    compare_queries(query1, query2, workspace, context)


@app.command()
def diagnose(
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to diagnose")
) -> None:
    """Diagnose workspace issues and show detailed status."""
    from sqlgenx.cli.commands.diagnose import diagnose_workspace
    diagnose_workspace(workspace)


@app.command()
def repair(
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to repair"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
) -> None:
    """Repair workspace by rebuilding embeddings."""
    from sqlgenx.cli.commands.diagnose import repair_workspace
    repair_workspace(workspace, force)


if __name__ == "__main__":
    app()