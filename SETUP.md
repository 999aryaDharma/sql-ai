# SQLGenX Setup Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Google Gemini API key

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sqlgenx.git
cd sqlgenx
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Development installation (editable mode)
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### 4. Configure API Key

Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

**Option 1: Environment Variable**
```bash
export GEMINI_API_KEY='your-api-key-here'
```

**Option 2: .env File**
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 5. Verify Installation

```bash
# Check if SQLGenX is installed
sqlgenx --help

# Should display the help menu with available commands
```

## Project Structure Setup

After installation, your project structure should look like:

```
sqlgenx/
├── sqlgenx/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── commands/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schema_loader.py
│   │   ├── vector_store.py
│   │   ├── llm_engine.py
│   │   ├── validator.py
│   │   └── graph.py
│   └── utils/
│       ├── __init__.py
│       ├── config_manager.py
│       ├── workspace_manager.py
│       └── rich_helpers.py
├── tests/
├── examples/
├── pyproject.toml
├── README.md
└── .env
```

## First Time Usage

### 1. Load Your First Schema

```bash
# Use the example schema
sqlgenx load examples/ecommerce_schema.sql --name ecommerce --dbms mysql

# Or use your own schema
sqlgenx load /path/to/your/schema.sql --name my_db --dbms postgresql
```

### 2. Generate Your First Query

```bash
sqlgenx generate "Show me all customers"
```

### 3. List Workspaces

```bash
sqlgenx list
```

## Workspace Directory

SQLGenX stores all data in `~/.sqlgenx/`:

```
~/.sqlgenx/
├── config.json           # Global configuration
└── workspaces/
    ├── ecommerce/
    │   ├── schema.sql    # Your schema file
    │   ├── embeddings/   # ChromaDB vector store
    │   ├── history/      # Query history (future)
    │   └── meta.json     # Workspace metadata
    └── another_db/
        └── ...
```

## Troubleshooting

### "Command not found: sqlgenx"

Make sure you've installed the package and activated your virtual environment:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

### "API key not configured"

Set your Gemini API key:
```bash
export GEMINI_API_KEY='your-key'
# Or create a .env file
```

### "Failed to parse schema"

- Check if your SQL file is valid
- Try specifying the correct DBMS type: `--dbms mysql` or `--dbms postgresql`
- Check for syntax errors in your schema

### Import Errors

If you get import errors, reinstall dependencies:
```bash
pip install -e . --force-reinstall
```

## Development Setup

For contributors:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=sqlgenx --cov-report=html

# Format code
black sqlgenx/

# Lint code
ruff check sqlgenx/

# Type checking
mypy sqlgenx/
```

## Updating

To get the latest version:

```bash
git pull origin main
pip install -e . --upgrade
```

## Uninstallation

```bash
pip uninstall sqlgenx

# Remove workspace directory if needed
rm -rf ~/.sqlgenx
```

## Next Steps

- Read the [README.md](README.md) for usage examples
- Check the [examples/](examples/) directory for sample schemas
- Join our community for support and updates

## Support

If you encounter any issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/yourusername/sqlgenx/issues)
3. Create a new issue with detailed information

Happy SQL generating! 🚀