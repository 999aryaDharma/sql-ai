"""Debug script to see exactly how schema context is formatted for LLM planning."""
import os
from pathlib import Path
from sqlgenx.core.schema_loader import SchemaLoader
from sqlgenx.core.vector_store import VectorStore
from sqlgenx.core.llm_engine import LLMEngine
import tempfile

def debug_schema_context():
    """Show exactly how schema context is formatted and sent to LLM for planning."""
    
    # Find the workspace directory
    home_dir = Path.home()
    workspace_dir = home_dir / ".sqlgenx" / "workspaces"
    
    # Look for available workspaces
    if not workspace_dir.exists():
        print(f"Workspace directory does not exist: {workspace_dir}")
        return
    
    workspaces = [d for d in workspace_dir.iterdir() if d.is_dir()]
    if not workspaces:
        print(f"No workspaces found in {workspace_dir}")
        print("Available directories:")
        for item in workspace_dir.iterdir():
            print(f"  - {item.name}")
        return
    
    print("Available workspaces:")
    for i, ws in enumerate(workspaces):
        print(f"  {i+1}. {ws.name}")
    
    # Ask user to select a workspace
    print(f"\nFound {len(workspaces)} workspace(s).")
    if len(workspaces) == 1:
        selected_index = 0
        print(f"Using only workspace: {workspaces[selected_index].name}")
    else:
        while True:
            try:
                selected_index = int(input(f"Enter workspace number (1-{len(workspaces)}): ")) - 1
                if 0 <= selected_index < len(workspaces):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(workspaces)}")
            except ValueError:
                print("Please enter a valid number")
    
    selected_workspace = workspaces[selected_index]
    print(f"\nUsing workspace: {selected_workspace.name}")
    
    schema_file = selected_workspace / "schema.sql"
    if not schema_file.exists():
        print(f"Schema file not found: {schema_file}")
        return
    
    print(f"\nLoading schema from: {schema_file}")
    
    # Load schema using our enhanced loader
    loader = SchemaLoader(dialect="")  # Generic dialect
    schema_info = loader.load_from_file(schema_file)
    
    print("\n" + "="*60)
    print("SCHEMA LOADED BY SCHEMA LOADER (Enhanced with relationships):")
    print("="*60)
    
    for table in schema_info.tables:
        print(f"\n{table.to_text()}")
    
    print("\n" + "="*60)
    print("SCHEMA CONTEXT AS SENT TO LLM FOR PLANNING (NATURAL LANGUAGE FORMAT):")
    print("="*60)
    
    # Use the new natural language format
    schema_context = schema_info.to_natural_language_schema()
    
    print("Complete schema context sent to LLM (Natural Language Format):")
    print("-" * 40)
    print(schema_context)
    print("-" * 40)
    
    # Show how this context would be used in the planning prompt
    test_user_query = "Find products where quantity sold this week dropped more than 40% compared to last week."
    dbms_type = "generic"
    
    print(f"\n" + "="*60)
    print("FULL PLANNING PROMPT THAT WOULD BE SENT TO LLM:")
    print("="*60)
    
    planning_prompt = f"""You are an expert SQL engineer with deep understanding of database schemas.

Before creating the execution plan, carefully analyze the schema and identify both available capabilities and potential limitations.

Schema Context:
{schema_context}

Database Type: {dbms_type}

User Request: "{test_user_query}"

CRITICAL: Before creating the plan, do these checks:
1. Identify all required entities (tables, columns, date fields, etc.) from the user request
2. Verify if these entities exist in the schema directly (in the same table) OR if they can be accessed through relationships/joins
3. Check for date filtering capabilities - can required dates be accessed through joins if not directly available?
4. Map out table relationships - which tables can be joined to get required data?
5. Pay special attention to temporal columns and how they can be accessed through joins
6. Only mark as impossible if there's truly NO PATH to access required data

For example: If transaction_item doesn't have a date column but is linked to a transaction table that has a date, this is possible through JOIN.
Do NOT assume missing functionality if it can be achieved through table relationships.

If the data truly cannot be accessed due to schema limitations, return "VALIDATION_FAILED: [specific explanation of what's missing and why no alternative path exists]"

If the query is possible, then create a detailed execution plan considering:
1. Tables involved and their relationships
2. Required joins and their conditions (be explicit about how to connect tables)
3. Filters and where clauses (including date filters that may come from related tables)
4. Aggregations and groupings needed
5. Any date range considerations (including how vector stores retrieve context)
6. How to properly group/aggregate across related tables

Provide only the execution plan if possible, or the validation failure message:"""
    
    print("Planning Prompt:")
    print("-" * 40)
    print(planning_prompt)
    print("-" * 40)
    
    print(f"\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. The schema context includes relationship information (Foreign Keys, Referenced by)")
    print("2. Temporal columns are explicitly marked (TEMPORAL)")
    print("3. This should help the LLM understand that date filtering can be done through JOINs")
    print("4. For example, if 'transaction_items' links to 'transactions' which has dates, this is possible")
    print("5. The LLM is specifically instructed not to assume impossibility without checking relationships")

if __name__ == "__main__":
    debug_schema_context()