"""Configuration manager for SQLGenX."""
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration."""
    
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    default_workspace: str = Field(default="default")
    sqlgenx_home: Path = Field(default=Path.home() / ".sqlgenx")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class WorkspaceMeta(BaseModel):
    """Metadata for a workspace."""
    
    name: str
    schema_path: str
    dbms_type: str = "generic"
    created_at: str
    last_used: str
    description: Optional[str] = None


class ConfigManager:
    """Manages application configuration and settings."""
    
    def __init__(self) -> None:
        self.config = Config()
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.config.sqlgenx_home.mkdir(parents=True, exist_ok=True)
        (self.config.sqlgenx_home / "workspaces").mkdir(exist_ok=True)
        (self.config.sqlgenx_home / "config.json").touch(exist_ok=True)
    
    def get_workspace_dir(self, workspace_name: str) -> Path:
        """Get the directory path for a workspace."""
        return self.config.sqlgenx_home / "workspaces" / workspace_name
    
    def save_global_config(self, key: str, value: str) -> None:
        """Save a configuration value."""
        config_file = self.config.sqlgenx_home / "config.json"
        
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config_data = {}
        
        config_data[key] = value
        
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
    
    def get_global_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a configuration value."""
        config_file = self.config.sqlgenx_home / "config.json"
        
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
            return config_data.get(key, default)
        except (FileNotFoundError, json.JSONDecodeError):
            return default
    
    def get_current_workspace(self) -> str:
        """Get the current active workspace."""
        return self.get_global_config("current_workspace", self.config.default_workspace) or self.config.default_workspace
    
    def set_current_workspace(self, workspace_name: str) -> None:
        """Set the current active workspace."""
        self.save_global_config("current_workspace", workspace_name)