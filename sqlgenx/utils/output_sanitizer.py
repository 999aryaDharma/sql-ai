"""Enhanced output sanitizer for cleaning LLM responses."""
import re
from typing import Optional


class OutputSanitizer:
    """Clean and sanitize all LLM outputs to remove unwanted characters."""
    
    # ANSI escape sequences (colors, formatting)
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    # Emoji and special unicode characters
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    
    # Markdown code blocks
    CODE_BLOCK = re.compile(r'```(?:sql|SQL)?\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)
    
    # Unwanted prefixes that LLMs often add
    UNWANTED_PREFIXES = [
        r'^Here is the SQL query:?\s*',
        r'^Here\'s the SQL query:?\s*',
        r'^SQL Query:?\s*',
        r'^Query:?\s*',
        r'^The SQL query is:?\s*',
        r'^\*\*SQL Query:\*\*\s*',
        r'^Let me generate:?\s*',
        r'^I\'ll generate:?\s*',
        r'^Sure,?\s*',
        r'^Certainly,?\s*',
    ]
    
    # SQL keywords for detection
    SQL_KEYWORDS = {
        'SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE',
        'ALTER', 'DROP', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT',
        'INNER', 'OUTER', 'ON', 'GROUP', 'BY', 'HAVING', 'ORDER',
        'LIMIT', 'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT', 'AS'
    }
    
    @classmethod
    def clean_sql(cls, raw_sql: str) -> str:
        """
        Clean SQL output from LLM completely.
        
        Args:
            raw_sql: Raw SQL string from LLM
            
        Returns:
            Cleaned SQL string ready for execution
        """
        if not raw_sql or not raw_sql.strip():
            return ""
        
        cleaned = raw_sql.strip()
        
        # Step 1: Remove ANSI escape codes
        cleaned = cls.ANSI_ESCAPE.sub('', cleaned)
        
        # Step 2: Remove emoji and special unicode
        cleaned = cls.EMOJI_PATTERN.sub('', cleaned)
        
        # Step 3: Extract from markdown code blocks if present
        match = cls.CODE_BLOCK.search(cleaned)
        if match:
            cleaned = match.group(1).strip()
        
        # Step 4: Remove inline backticks
        cleaned = cleaned.replace('`', '')
        
        # Step 5: Remove unwanted prefixes
        for prefix in cls.UNWANTED_PREFIXES:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Step 6: Extract only SQL content (stop at explanatory text)
        cleaned = cls._extract_sql_content(cleaned)
        
        # Step 7: Clean up whitespace
        cleaned = cls._normalize_whitespace(cleaned)
        
        # Step 8: Ensure proper semicolon
        if cleaned and not cleaned.rstrip().endswith(';'):
            cleaned = cleaned.rstrip() + ';'
        
        # Step 9: Remove duplicate semicolons
        cleaned = re.sub(r';+\s*$', ';', cleaned)
        
        return cleaned
    
    @classmethod
    def _extract_sql_content(cls, text: str) -> str:
        """Extract only SQL content, removing explanatory text."""
        lines = text.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Check if line starts with SQL keyword
            starts_with_sql = any(
                line_upper.startswith(kw) for kw in cls.SQL_KEYWORDS
            )
            
            # Check if line contains SQL-like content
            has_sql_content = any(kw in line_upper for kw in cls.SQL_KEYWORDS)
            
            # Start capturing SQL
            if starts_with_sql:
                in_sql = True
            
            # Capture SQL lines
            if in_sql:
                # Skip empty lines at the start
                if not sql_lines and not line_stripped:
                    continue
                
                sql_lines.append(line)
                
                # Stop at semicolon
                if line_stripped.endswith(';'):
                    break
                    
            # Stop if we hit non-SQL text after SQL started
            elif in_sql and line_stripped and not has_sql_content:
                # Check if it's a SQL comment
                if not (line_stripped.startswith('--') or line_stripped.startswith('/*')):
                    break
        
        return '\n'.join(sql_lines).strip()
    
    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        """Normalize whitespace in SQL."""
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove excessive empty lines (max 1 empty line between content)
        normalized = []
        prev_empty = False
        
        for line in lines:
            is_empty = not line.strip()
            
            if is_empty:
                if not prev_empty:
                    normalized.append(line)
                prev_empty = True
            else:
                normalized.append(line)
                prev_empty = False
        
        return '\n'.join(normalized).strip()
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Clean general text output (explanations, analysis).
        
        Args:
            text: Raw text from LLM
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        cleaned = text.strip()
        
        # Remove ANSI codes
        cleaned = cls.ANSI_ESCAPE.sub('', cleaned)
        
        # Remove emoji
        cleaned = cls.EMOJI_PATTERN.sub('', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 newlines
        cleaned = re.sub(r' +', ' ', cleaned)  # Single spaces
        
        # Remove common LLM artifacts
        cleaned = re.sub(r'\*\*\*+', '', cleaned)  # Multiple asterisks
        
        return cleaned.strip()
    
    @classmethod
    def clean_explanation(cls, text: str) -> str:
        """
        Clean explanation text, removing common prefixes.
        
        Args:
            text: Raw explanation from LLM
            
        Returns:
            Cleaned explanation
        """
        cleaned = cls.clean_text(text)
        
        # Remove common explanation prefixes
        prefixes = [
            r'^Let\'s break it down:?\s*',
            r'^Here\'s the explanation:?\s*',
            r'^Explanation:?\s*',
            r'^This query:?\s*',
        ]
        
        for prefix in prefixes:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    @classmethod
    def is_valid_sql(cls, sql: str) -> bool:
        """
        Quick validation if string looks like SQL.
        
        Args:
            sql: SQL string to validate
            
        Returns:
            True if looks like valid SQL
        """
        if not sql or not sql.strip():
            return False
        
        sql_upper = sql.upper()
        
        # Must contain at least one major SQL keyword
        major_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH']
        
        return any(kw in sql_upper for kw in major_keywords)
    
    @classmethod
    def extract_code_block(cls, text: str, language: str = 'sql') -> Optional[str]:
        """
        Extract code block from markdown.
        
        Args:
            text: Text containing code block
            language: Language identifier (default: 'sql')
            
        Returns:
            Extracted code or None
        """
        pattern = rf'```{language}\s*(.*?)\s*```'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Try without language specifier
        pattern = r'```\s*(.*?)\s*```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return None