# SQLGenX

AI-powered CLI for Smart SQL Generation using LLM and Vector Search

```
███████╗ ██████╗ ██╗      ██████╗ ███████╗███╗   ██╗██╗  ██╗
██╔════╝██╔═══██╗██║     ██╔════╝ ██╔════╝████╗  ██║╚██╗██╔╝
███████╗██║   ██║██║     ██║  ███╗█████╗  ██╔██╗ ██║ ╚███╔╝ 
╚════██║██║▄▄ ██║██║     ██║   ██║██╔══╝  ██║╚██╗██║ ██╔██╗ 
███████║╚██████╔╝███████╗╚██████╔╝███████╗██║ ╚████║██╔╝ ██╗
╚══════╝ ╚══▀▀═╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝
```

## 🚀 Features

- **Schema Awareness**: Load and understand your database schema
- **Natural Language to SQL**: Convert plain English to SQL queries
- **Multi-Database Support**: MySQL, PostgreSQL, SQLite, and more
- **Workspace Management**: Manage multiple database schemas
- **Smart Context Retrieval**: Uses vector embeddings for relevant schema context
- **SQL Validation**: Automatic syntax and structure validation
- **Beautiful CLI**: Rich terminal output with syntax highlighting

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sqlgenx.git
cd sqlgenx

# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install sqlgenx
```

## 🔑 Setup

1. Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Set your API key:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

## 🎯 Quick Start

### 1. Load a Schema

```bash
# Load a SQL schema file
sqlgenx load ./schema.sql --name my_database --dbms postgresql

# With description
sqlgenx load ./hr_schema.sql --name hr_db --dbms mysql --desc "HR Management Database"
```

### 2. Generate SQL Queries

```bash
# Generate SQL from natural language
sqlgenx generate "Show me the top 5 employees with highest salary"

# Generate with explanation
sqlgenx generate "Total sales by region this year" --explain

# Use a specific workspace
sqlgenx generate "List all active customers" --workspace sales_db
```

### 3. Manage Workspaces

```bash
# List all workspaces
sqlgenx list

# Switch workspace
sqlgenx use finance_db

# Show workspace info
sqlgenx info

# Delete a workspace
sqlgenx delete old_db
```

## 📖 Commands

### `sqlgenx load`
Load a SQL schema file into a workspace.

**Options:**
- `--name, -n`: Workspace name (default: filename)
- `--dbms, -d`: Database type (mysql, postgresql, sqlite, generic)
- `--desc`: Workspace description

### `sqlgenx generate`
Generate SQL from natural language query.

**Options:**
- `--workspace, -w`: Use specific workspace
- `--explain, -e`: Generate explanation for the query
- `--copy, -c`: Copy SQL to clipboard

### `sqlgenx use`
Switch to a different workspace.

### `sqlgenx list`
List all available workspaces.

### `sqlgenx info`
Show information about current or specified workspace.

### `sqlgenx config`
Configure SQLGenX settings.

**Options:**
- `--show, -s`: Show current configuration

### `sqlgenx delete`
Delete a workspace.

**Options:**
- `--force, -f`: Skip confirmation

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         CLI Interface (Typer)       │
│  User commands & interaction        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      LangGraph Workflow Engine      │
│  1. Load Schema                     │
│  2. Create Embeddings (Chroma)      │
│  3. Retrieve Context (Vector Search)│
│  4. Generate SQL (Gemini LLM)       │
│  5. Validate SQL (SQLGlot)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     Workspace Storage (~/.sqlgenx)  │
│  ├─ workspaces/                     │
│  │  ├─ db_name/                     │
│  │  │  ├─ schema.sql                │
│  │  │  ├─ embeddings/               │
│  │  │  ├─ history/                  │
│  │  │  └─ meta.json                 │
└─────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **CLI Framework**: Typer
- **LLM**: Google Gemini (via LangChain)
- **Vector Database**: ChromaDB
- **SQL Parser**: SQLGlot
- **Orchestration**: LangGraph
- **UI**: Rich

## 📁 Project Structure

```
sqlgenx/
├── cli/
│   ├── main.py              # CLI entry point
│   └── commands/            # Command implementations
│       ├── load.py
│       ├── generate.py
│       ├── switch.py
│       └── config.py
├── core/
│   ├── schema_loader.py     # Schema parsing
│   ├── vector_store.py      # ChromaDB integration
│   ├── llm_engine.py        # LangChain + Gemini
│   ├── validator.py         # SQL validation
│   └── graph.py             # LangGraph workflow
├── utils/
│   ├── config_manager.py    # Configuration
│   ├── workspace_manager.py # Workspace management
│   └── rich_helpers.py      # CLI formatting
└── tests/                   # Test suite
```

## 🧪 Examples

### Example 1: E-commerce Database

```bash
# Load schema
sqlgenx load ecommerce.sql --name shop --dbms postgresql

# Generate queries
sqlgenx generate "Top 10 products by revenue this month"
sqlgenx generate "Customers who haven't ordered in 90 days"
sqlgenx generate "Average order value by customer segment"
```

### Example 2: HR Database

```bash
# Load schema
sqlgenx load hr_schema.sql --name hr --dbms mysql

# Generate queries
sqlgenx generate "Employees in engineering department hired this year"
sqlgenx generate "Department with highest average salary"
sqlgenx generate "Employee count by location and role"
```

## 🔮 Future Enhancements

- [ ] Interactive chat mode (`sqlgenx chat`)
- [ ] Live database connection support
- [ ] Query history and favorites
- [ ] Natural language query validation
- [ ] VSCode and DBeaver plugins
- [ ] Multi-query generation
- [ ] Query optimization suggestions

## 🙏 Acknowledgments

- Google Gemini for powerful LLM capabilities
- LangChain for LLM orchestration
- ChromaDB for vector search
- SQLGlot for SQL parsing