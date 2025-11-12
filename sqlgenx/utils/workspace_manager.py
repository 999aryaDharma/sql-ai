"""Workspace manager for SQLGenX."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class WorkspaceMeta:
    """Metadata for a workspace."""
    
    def __init__(
        self,
        name: str,
        schema_path: str,
        dbms_type: str = "generic",
        created_at: str = "",
        last_used: str = "",
        description: Optional[str] = None
    ):
        self.name = name
        self.schema_path = schema_path
        self.dbms_type = dbms_type
        self.created_at = created_at or datetime.now().isoformat()
        self.last_used = last_used or datetime.now().isoformat()
        self.description = description
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "schema_path": self.schema_path,
            "dbms_type": self.dbms_type,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceMeta":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            schema_path=data.get("schema_path", ""),
            dbms_type=data.get("dbms_type", "generic"),
            created_at=data.get("created_at", ""),
            last_used=data.get("last_used", ""),
            description=data.get("description")
        )


class ConfigManager:
    """Manages application configuration and settings."""
    
    def __init__(self) -> None:
        from pathlib import Path
        import os
        from dotenv import load_dotenv
        
        # Load .env file
        load_dotenv()
        
        self.sqlgenx_home = Path.home() / ".sqlgenx"
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.default_workspace = "default"
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.sqlgenx_home.mkdir(parents=True, exist_ok=True)
        (self.sqlgenx_home / "workspaces").mkdir(exist_ok=True)
        config_file = self.sqlgenx_home / "config.json"
        if not config_file.exists():
            config_file.write_text("{}")
    
    def get_workspace_dir(self, workspace_name: str) -> Path:
        """Get the directory path for a workspace."""
        return self.sqlgenx_home / "workspaces" / workspace_name
    
    def save_global_config(self, key: str, value: str) -> None:
        """Save a configuration value."""
        config_file = self.sqlgenx_home / "config.json"
        
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
        config_file = self.sqlgenx_home / "config.json"
        
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
            return config_data.get(key, default)
        except (FileNotFoundError, json.JSONDecodeError):
            return default
    
    def get_current_workspace(self) -> str:
        """Get the current active workspace."""
        return self.get_global_config("current_workspace", self.default_workspace) or self.default_workspace
    
    def set_current_workspace(self, workspace_name: str) -> None:
        """Set the current active workspace."""
        self.save_global_config("current_workspace", workspace_name)


class WorkspaceManager:
    """Manages workspaces for different database schemas."""
    
    def __init__(self) -> None:
        self.config_manager = ConfigManager()
    
    def create_workspace(
        self, 
        name: str, 
        schema_path: str,
        dbms_type: str = "generic",
        description: Optional[str] = None
    ) -> Path:
        """Create a new workspace."""
        workspace_dir = self.config_manager.get_workspace_dir(name)
        
        if workspace_dir.exists():
            raise ValueError(f"Workspace '{name}' already exists")
        
        # Create workspace structure
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "embeddings").mkdir(exist_ok=True)
        (workspace_dir / "history").mkdir(exist_ok=True)
        
        # Copy schema file
        schema_dest = workspace_dir / "schema.sql"
        shutil.copy2(schema_path, schema_dest)
        
        # Create metadata
        now = datetime.now().isoformat()
        meta = WorkspaceMeta(
            name=name,
            schema_path=str(schema_dest),
            dbms_type=dbms_type,
            created_at=now,
            last_used=now,
            description=description
        )
        
        self._save_meta(workspace_dir, meta)
        
        return workspace_dir
    
    def get_workspace(self, name: str) -> Optional[Path]:
        """Get workspace directory if it exists."""
        workspace_dir = self.config_manager.get_workspace_dir(name)
        return workspace_dir if workspace_dir.exists() else None
    
    def list_workspaces(self) -> List[WorkspaceMeta]:
        """List all available workspaces."""
        workspaces_dir = self.config_manager.sqlgenx_home / "workspaces"
        workspaces = []
        
        if not workspaces_dir.exists():
            return workspaces
        
        for workspace_path in workspaces_dir.iterdir():
            if workspace_path.is_dir():
                meta = self._load_meta(workspace_path)
                if meta:
                    workspaces.append(meta)
        
        return workspaces
    
    def switch_workspace(self, name: str) -> None:
        """Switch to a different workspace."""
        workspace_dir = self.get_workspace(name)
        
        if not workspace_dir:
            raise ValueError(f"Workspace '{name}' does not exist")
        
        # Update last used time
        meta = self._load_meta(workspace_dir)
        if meta:
            meta.last_used = datetime.now().isoformat()
            self._save_meta(workspace_dir, meta)
        
        self.config_manager.set_current_workspace(name)
    
    def delete_workspace(self, name: str) -> None:
        """Delete a workspace."""
        workspace_dir = self.get_workspace(name)
        
        if not workspace_dir:
            raise ValueError(f"Workspace '{name}' does not exist")
        
        current_workspace = self.config_manager.get_current_workspace()
        if current_workspace == name:
            # Jika workspace yang akan dihapus adalah workspace aktif saat ini,
            # hapus dan atur ulang workspace aktif ke default.
            shutil.rmtree(workspace_dir)
            self.config_manager.set_current_workspace(self.config_manager.default_workspace)
            return True # Menunjukkan bahwa workspace aktif telah dihapus
        
        shutil.rmtree(workspace_dir)
        return False # Menunjukkan bahwa workspace non-aktif telah dihapus
    
    def get_current_workspace_meta(self) -> Optional[WorkspaceMeta]:
        """Get metadata for the current workspace."""
        current_name = self.config_manager.get_current_workspace()
        workspace_dir = self.get_workspace(current_name)
        
        if not workspace_dir:
            return None
        
        return self._load_meta(workspace_dir)
    
    def _save_meta(self, workspace_dir: Path, meta: WorkspaceMeta) -> None:
        """Save workspace metadata."""
        meta_file = workspace_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)
    
    def _load_meta(self, workspace_dir: Path) -> Optional[WorkspaceMeta]:
        """Load workspace metadata."""
        meta_file = workspace_dir / "meta.json"
        
        if not meta_file.exists():
            return None
        
        try:
            with open(meta_file, "r") as f:
                data = json.load(f)
            return WorkspaceMeta.from_dict(data)
        except (json.JSONDecodeError, ValueError):
            return None