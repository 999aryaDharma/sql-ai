"""SQL validator using SQLGlot."""
from typing import Dict, List, Tuple, Optional
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class ValidationResult:
    """Result of SQL validation."""
    
    def __init__(
        self, 
        is_valid: bool, 
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        parsed_query: Optional[exp.Expression] = None
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.parsed_query = parsed_query
    
    def __bool__(self) -> bool:
        return self.is_valid


class SQLValidator:
    """Validates SQL queries for syntax and structure."""
    
    def __init__(self, dialect: str = ""):
        """Initialize validator with optional dialect."""
        self.dialect = dialect or None
    
    def validate(self, sql: str) -> ValidationResult:
        """Validate SQL query."""
        errors = []
        warnings = []
        parsed = None
        
        # Check if SQL is empty
        if not sql or not sql.strip():
            errors.append("SQL query is empty")
            return ValidationResult(False, errors, warnings)
        
        # Try to parse with SQLGlot
        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            
            # Additional checks
            self._check_query_structure(parsed, warnings)
            
        except ParseError as e:
            errors.append(f"Syntax error: {str(e)}")
            return ValidationResult(False, errors, warnings)
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, parsed)
    
    def _check_query_structure(self, parsed: exp.Expression, warnings: List[str]) -> None:
        """Perform structural checks on parsed query."""
        
        # Check for SELECT * without LIMIT (potential performance issue)
        if isinstance(parsed, exp.Select):
            has_star = any(isinstance(expr, exp.Star) for expr in parsed.expressions)
            has_limit = parsed.args.get("limit") is not None
            
            if has_star and not has_limit:
                warnings.append(
                    "Query uses SELECT * without LIMIT - this might return many rows"
                )
            
            # Check for missing GROUP BY with aggregate functions
            has_aggregates = self._has_aggregates(parsed)
            has_group_by = parsed.args.get("group") is not None
            
            if has_aggregates and not has_group_by:
                non_aggregate_cols = self._get_non_aggregate_columns(parsed)
                if non_aggregate_cols:
                    warnings.append(
                        f"Query has aggregate functions but no GROUP BY. "
                        f"Consider grouping by: {', '.join(non_aggregate_cols)}"
                    )
    
    def _has_aggregates(self, node: exp.Expression) -> bool:
        """Check if query contains aggregate functions."""
        aggregate_funcs = {"COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT"}
        
        for expr in node.walk():
            if isinstance(expr, exp.Func):
                func_name = expr.sql_name().upper()
                if func_name in aggregate_funcs:
                    return True
        
        return False
    
    def _get_non_aggregate_columns(self, select: exp.Select) -> List[str]:
        """Get column names that are not wrapped in aggregate functions."""
        columns = []
        
        for expr in select.expressions:
            if isinstance(expr, exp.Column):
                columns.append(expr.name)
        
        return columns
    
    def format_sql(self, sql: str, pretty: bool = True) -> str:
        """Format SQL query for better readability."""
        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            return parsed.sql(dialect=self.dialect, pretty=pretty)
        except Exception:
            # If formatting fails, return original
            return sql
    
    def get_tables_used(self, sql: str) -> List[str]:
        """Extract table names used in the query."""
        try:
            parsed = sqlglot.parse_one(sql, read=self.dialect)
            tables = set()
            cte_names = set()

            # First, find all CTE names to ignore them later
            for cte in parsed.find_all(exp.CTE):
                cte_names.add(cte.alias.lower())
            
            for table in parsed.find_all(exp.Table):
                # Only add the table if it's not a CTE
                if table.name.lower() not in cte_names:
                    tables.add(table.name)
            
            return sorted(list(tables))
        except Exception:
            return []
    
    def check_schema_compatibility(
        self, 
        sql: str, 
        available_tables: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check if query only uses tables that exist in schema."""
        used_tables = self.get_tables_used(sql)
        available_set = set(t.lower() for t in available_tables)
        
        missing_tables = []
        for table in used_tables:
            if table.lower() not in available_set:
                missing_tables.append(table)
        
        is_compatible = len(missing_tables) == 0
        return is_compatible, missing_tables