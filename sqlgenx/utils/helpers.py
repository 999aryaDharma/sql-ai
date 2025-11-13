"""General helper functions for SQLGenX."""

# Kamus ini adalah "Single Source of Truth" untuk nama dialek.
# Kunci adalah input umum dari pengguna (dalam huruf kecil).
# Nilai adalah nama dialek yang dikenali oleh SQLGlot.
DIALECT_MAP = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "mysql": "mysql",
    "mariadb": "mysql",  # MariaDB kompatibel dengan dialek MySQL di SQLGlot
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "sql server": "tsql",
    "mssql": "tsql",
}


def normalize_dialect(dialect: str) -> str:
    """Normalize database dialect names to be compatible with SQLGlot."""
    if not dialect:
        return "generic"
    return DIALECT_MAP.get(dialect.lower(), dialect)