"""Main CLI entry point for SQLGenX."""
import typer
from pathlib import Path
from typing import Optional
from sqlgenx.utils.rich_helpers import print_banner, console

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
        console.print("\nRun [bold]'sqlgenx --help'[/bold] to see commands, or [bold]'sqlgenx [COMMAND] --help'[/bold] for details on a specific command.")


@app.command()
def load(
    schema_file: Path = typer.Argument(..., help="Path to SQL schema file"),
    name: str = typer.Option(None, "--name", "-n", help="Name for the workspace. Defaults to the schema filename."),
    dbms: str = typer.Option("generic", "--dbms", "-d", help="Database type (e.g., mysql, postgresql). Defaults to 'generic'."),
    description: str = typer.Option(None, "--desc", help="A brief description for the workspace.")
) -> None:
    """Creates a new workspace from a SQL schema file, generating vector embeddings for context-aware queries."""
    from sqlgenx.cli.commands.load import load_schema
    load_schema(schema_file, name, dbms, description)


@app.command()
def generate(
    query: str = typer.Argument(..., help="Natural language query"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Specify a workspace to use, overriding the active one."),
    explain: bool = typer.Option(False, "--explain", "-e", help="Provide a natural language explanation of the generated SQL query."),
    copy: bool = typer.Option(False, "--copy", "-c", help="Copy the generated SQL query to the clipboard.")
) -> None:
    """Generates a SQL query from a natural language prompt using the active workspace's schema."""
    from sqlgenx.cli.commands.generate import generate_sql
    generate_sql(query, workspace, explain, copy)


@app.command()
def use(
    workspace: str = typer.Argument(..., help="Workspace name to switch to")
) -> None:
    """Sets a workspace as the active one for future commands."""
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
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without asking for confirmation.")
) -> None:
    """Deletes a workspace and all its associated data."""
    from sqlgenx.cli.commands.switch import delete_workspace_cmd
    delete_workspace_cmd(workspace, force)


@app.command()
def config(
    key: str = typer.Argument(None, help="Configuration key to set (e.g., 'current_workspace')."),
    value: str = typer.Argument(None, help="Value to set for the configuration key."),
    show: bool = typer.Option(False, "--show", "-s", help="Display the current global configuration.")
) -> None:
    """Manages global configuration settings for SQLGenX."""
    from sqlgenx.cli.commands.config import config_cmd
    config_cmd(key, value, show)


@app.command()
def info(
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace to show info for")
) -> None:
    """Displays detailed information about a workspace, including its tables."""
    from sqlgenx.cli.commands.config import info_cmd
    info_cmd(workspace)


if __name__ == "__main__":
    app()