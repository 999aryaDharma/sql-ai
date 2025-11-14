"""Persistent cache manager for LLM responses."""
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from functools import wraps


class CacheManager:
    """
    Persistent file-based cache manager for LLM responses.
    Reduces API calls by caching frequently used queries.
    """
    
    def __init__(
        self, 
        workspace_dir: Path,
        default_ttl: int = 3600,  # 1 hour default
        max_cache_size: int = 100  # Max entries per cache type
    ):
        """
        Initialize cache manager.
        
        Args:
            workspace_dir: Workspace directory for cache storage
            default_ttl: Default time-to-live in seconds
            max_cache_size: Maximum number of cache entries
        """
        self.cache_dir = workspace_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        
        # Separate cache files for different operations
        self.sql_cache_file = self.cache_dir / "sql_generation.json"
        self.validation_cache_file = self.cache_dir / "validation.json"
        self.analysis_cache_file = self.cache_dir / "analysis.json"
        self.optimization_cache_file = self.cache_dir / "optimization.json"
        
        # Load existing caches
        self.sql_cache = self._load_cache(self.sql_cache_file)
        self.validation_cache = self._load_cache(self.validation_cache_file)
        self.analysis_cache = self._load_cache(self.analysis_cache_file)
        self.optimization_cache = self._load_cache(self.optimization_cache_file)
        
        # Auto-cleanup expired entries on init
        self._cleanup_expired()
    
    def _generate_key(self, *args: Any) -> str:
        """Generate deterministic cache key from arguments."""
        # Convert all args to strings and join
        combined = "|".join(str(arg) for arg in args if arg is not None)
        
        # Generate SHA256 hash
        hash_object = hashlib.sha256(combined.encode('utf-8'))
        return hash_object.hexdigest()[:16]  # Use first 16 chars
    
    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """Load cache from JSON file."""
        if not cache_file.exists():
            return {}
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If cache is corrupted, start fresh
            return {}
    
    def _save_cache(self, cache_file: Path, cache_data: Dict[str, Any]) -> None:
        """Save cache to JSON file."""
        try:
            # Ensure directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with pretty printing for debugging
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            # Fail silently on cache write errors (cache is optional)
            print(f"Warning: Failed to save cache: {e}")
    
    def _is_expired(self, timestamp: str, ttl: Optional[int] = None) -> bool:
        """Check if cache entry is expired."""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            ttl_seconds = ttl or self.default_ttl
            age = datetime.now() - cached_time
            
            return age.total_seconds() > ttl_seconds
        except (ValueError, TypeError):
            # Invalid timestamp = expired
            return True
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from all caches."""
        caches = [
            (self.sql_cache, self.sql_cache_file),
            (self.validation_cache, self.validation_cache_file),
            (self.analysis_cache, self.analysis_cache_file),
            (self.optimization_cache, self.optimization_cache_file),
        ]
        
        for cache_data, cache_file in caches:
            original_size = len(cache_data)
            
            # Remove expired entries
            expired_keys = [
                k for k, v in cache_data.items()
                if self._is_expired(v.get('timestamp', ''), v.get('ttl'))
            ]
            
            for key in expired_keys:
                del cache_data[key]
            
            # Save if entries were removed
            if len(expired_keys) > 0:
                self._save_cache(cache_file, cache_data)
    
    def _enforce_size_limit(self, cache_data: Dict[str, Any]) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        if len(cache_data) <= self.max_cache_size:
            return
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            cache_data.items(),
            key=lambda x: x[1].get('timestamp', ''),
        )
        
        # Keep only the newest entries
        entries_to_keep = sorted_entries[-self.max_cache_size:]
        
        # Rebuild cache with only kept entries
        cache_data.clear()
        cache_data.update(dict(entries_to_keep))
    
    # ==================== SQL Generation Cache ====================
    
    def get_sql_generation(
        self,
        user_query: str,
        schema_context: str,
        dbms_type: str
    ) -> Optional[str]:
        """
        Get cached SQL generation result.
        
        Args:
            user_query: User's natural language query
            schema_context: Schema context string
            dbms_type: Database type
            
        Returns:
            Cached SQL or None if not found/expired
        """
        cache_key = self._generate_key(user_query, schema_context, dbms_type)
        
        if cache_key in self.sql_cache:
            entry = self.sql_cache[cache_key]
            
            if not self._is_expired(entry.get('timestamp', ''), entry.get('ttl')):
                # Update access time
                entry['last_accessed'] = datetime.now().isoformat()
                entry['access_count'] = entry.get('access_count', 0) + 1
                self._save_cache(self.sql_cache_file, self.sql_cache)
                
                return entry.get('sql')
            else:
                # Remove expired entry
                del self.sql_cache[cache_key]
                self._save_cache(self.sql_cache_file, self.sql_cache)
        
        return None
    
    def set_sql_generation(
        self,
        user_query: str,
        schema_context: str,
        dbms_type: str,
        sql: str,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache SQL generation result.
        
        Args:
            user_query: User's natural language query
            schema_context: Schema context string
            dbms_type: Database type
            sql: Generated SQL
            ttl: Optional custom TTL in seconds
        """
        cache_key = self._generate_key(user_query, schema_context, dbms_type)
        
        self.sql_cache[cache_key] = {
            'sql': sql,
            'timestamp': datetime.now().isoformat(),
            'ttl': ttl or self.default_ttl,
            'query': user_query,  # For debugging
            'dbms': dbms_type,
            'access_count': 0,
            'last_accessed': datetime.now().isoformat()
        }
        
        self._enforce_size_limit(self.sql_cache)
        self._save_cache(self.sql_cache_file, self.sql_cache)
    
    # ==================== Validation Cache ====================
    
    def get_validation(
        self,
        sql: str,
        dbms_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached validation result."""
        cache_key = self._generate_key(sql, dbms_type)
        
        if cache_key in self.validation_cache:
            entry = self.validation_cache[cache_key]
            
            if not self._is_expired(entry.get('timestamp', ''), entry.get('ttl')):
                return entry.get('result')
        
        return None
    
    def set_validation(
        self,
        sql: str,
        dbms_type: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """Cache validation result."""
        cache_key = self._generate_key(sql, dbms_type)
        
        self.validation_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'ttl': ttl or self.default_ttl
        }
        
        self._enforce_size_limit(self.validation_cache)
        self._save_cache(self.validation_cache_file, self.validation_cache)
    
    # ==================== Analysis Cache ====================
    
    def get_analysis(
        self,
        sql: str,
        data_hash: str
    ) -> Optional[str]:
        """Get cached data analysis result."""
        cache_key = self._generate_key(sql, data_hash)
        
        if cache_key in self.analysis_cache:
            entry = self.analysis_cache[cache_key]
            
            if not self._is_expired(entry.get('timestamp', ''), entry.get('ttl')):
                return entry.get('analysis')
        
        return None
    
    def set_analysis(
        self,
        sql: str,
        data_hash: str,
        analysis: str,
        ttl: Optional[int] = None
    ) -> None:
        """Cache data analysis result."""
        cache_key = self._generate_key(sql, data_hash)
        
        self.analysis_cache[cache_key] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'ttl': ttl or self.default_ttl
        }
        
        self._enforce_size_limit(self.analysis_cache)
        self._save_cache(self.analysis_cache_file, self.analysis_cache)
    
    # ==================== Optimization Cache ====================
    
    def get_optimization(
        self,
        sql: str,
        dbms_type: str
    ) -> Optional[str]:
        """Get cached optimization suggestions."""
        cache_key = self._generate_key(sql, dbms_type)
        
        if cache_key in self.optimization_cache:
            entry = self.optimization_cache[cache_key]
            
            if not self._is_expired(entry.get('timestamp', ''), entry.get('ttl')):
                return entry.get('suggestions')
        
        return None
    
    def set_optimization(
        self,
        sql: str,
        dbms_type: str,
        suggestions: str,
        ttl: Optional[int] = None
    ) -> None:
        """Cache optimization suggestions."""
        cache_key = self._generate_key(sql, dbms_type)
        
        self.optimization_cache[cache_key] = {
            'suggestions': suggestions,
            'timestamp': datetime.now().isoformat(),
            'ttl': ttl or self.default_ttl
        }
        
        self._enforce_size_limit(self.optimization_cache)
        self._save_cache(self.optimization_cache_file, self.optimization_cache)
    
    # ==================== Cache Management ====================
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        self.sql_cache.clear()
        self.validation_cache.clear()
        self.analysis_cache.clear()
        self.optimization_cache.clear()
        
        self._save_cache(self.sql_cache_file, self.sql_cache)
        self._save_cache(self.validation_cache_file, self.validation_cache)
        self._save_cache(self.analysis_cache_file, self.analysis_cache)
        self._save_cache(self.optimization_cache_file, self.optimization_cache)
    
    def clear_expired(self) -> int:
        """
        Clear all expired entries.
        
        Returns:
            Number of entries cleared
        """
        cleared = 0
        
        caches = [
            (self.sql_cache, self.sql_cache_file),
            (self.validation_cache, self.validation_cache_file),
            (self.analysis_cache, self.analysis_cache_file),
            (self.optimization_cache, self.optimization_cache_file),
        ]
        
        for cache_data, cache_file in caches:
            expired_keys = [
                k for k, v in cache_data.items()
                if self._is_expired(v.get('timestamp', ''), v.get('ttl'))
            ]
            
            for key in expired_keys:
                del cache_data[key]
                cleared += 1
            
            if expired_keys:
                self._save_cache(cache_file, cache_data)
        
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'sql_entries': len(self.sql_cache),
            'validation_entries': len(self.validation_cache),
            'analysis_entries': len(self.analysis_cache),
            'optimization_entries': len(self.optimization_cache),
            'total_entries': (
                len(self.sql_cache) +
                len(self.validation_cache) +
                len(self.analysis_cache) +
                len(self.optimization_cache)
            ),
            'cache_dir': str(self.cache_dir)
        }


def cached(ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds (uses default if not specified)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # This is a simple in-memory cache decorator
            # For file-based caching, use CacheManager directly
            
            if not hasattr(wrapper, '_cache'):
                wrapper._cache = {}
            
            # Generate cache key
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if cache_key in wrapper._cache:
                cached_value, timestamp = wrapper._cache[cache_key]
                age = time.time() - timestamp
                
                if ttl is None or age < ttl:
                    return cached_value
            
            # Call function and cache result
            result = func(self, *args, **kwargs)
            wrapper._cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator