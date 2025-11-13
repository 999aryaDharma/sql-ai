"""Rich console helpers for beautiful CLI output."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from typing import List, Optional
import sys

console = Console()


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]✗[/bold red] {message}", style="red")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}", style="yellow")


def print_sql(sql: str, title: str = "Generated SQL") -> None:
    """Print SQL code with syntax highlighting."""
    syntax = Syntax(sql, "sql", theme="monokai", line_numbers=True)
    panel = Panel(syntax, title=title, border_style="green")
    console.print(panel)


def print_workspaces(workspaces: List, current: str) -> None:
    """Print a table of workspaces."""
    if not workspaces:
        print_info("No workspaces found")
        return
    
    table = Table(title="Available Workspaces", show_header=True, header_style="bold violet")
    
    table.add_column("Active", style="green", width=6)
    table.add_column("Name", style="cyan")
    table.add_column("DBMS", style="yellow")
    table.add_column("Last Used", style="dim")
    table.add_column("Description", style="dim")
    
    for ws in workspaces:
        is_current = "  ✓" if ws.name == current else ""
        last_used = ws.last_used.split("T")[0] if "T" in ws.last_used else ws.last_used
        description = ws.description or "-"
        
        table.add_row(
            is_current,
            ws.name,
            ws.dbms_type,
            last_used,
            description[:50] + "..." if len(description) > 50 else description
        )
    
    console.print(table)


def print_banner() -> None:
    """Print SQLGenX banner."""
    banner = """
[bold cyan]
    ███████╗ ██████╗ ██╗      ██████╗ ███████╗███╗   ██╗██╗  ██╗
    ██╔════╝██╔═══██╗██║     ██╔════╝ ██╔════╝████╗  ██║╚██╗██╔╝
    ███████╗██║   ██║██║     ██║  ███╗█████╗  ██╔██╗ ██║ ╚███╔╝ 
    ╚════██║██║▄▄ ██║██║     ██║   ██║██╔══╝  ██║╚██╗██║ ██╔██╗ 
    ███████║╚██████╔╝███████╗╚██████╔╝███████╗██║ ╚████║██╔╝ ██╗
    ╚══════╝ ╚══▀▀═╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝
[/bold cyan]
[dim]    AI-powered CLI for Smart SQL Generation[/dim]
    """
    console.print(banner)


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    response = console.input(f"[bold yellow]?[/bold yellow] {message}{suffix}")
    
    if not response:
        return default
    
    return response.lower() in ["y", "yes"]


def print_table_info(table_name: str, columns: List[str]) -> None:
    """Print information about a table."""
    table = Table(title=f"Table: {table_name}", show_header=True)
    table.add_column("Column", style="cyan")
    
    for col in columns:
        table.add_row(col)
    
    console.print(table)


def print_context_info(contexts: List[dict]) -> None:
    """Print retrieved context information."""
    if not contexts:
        return
    
    console.print("\n[dim]Retrieved Schema Context:[/dim]")
    for ctx in contexts[:3]:  # Show top 3
        table_name = ctx.get("table", "unknown")
        score = ctx.get("relevance_score", 0.0)
        console.print(f"  • [cyan]{table_name}[/cyan] (relevance: {score:.2f})")