"""
Environment Variable Loader Utility

Loads environment variables from .env file if it exists.
This ensures consistent API key loading across all scripts.
"""

import os
from pathlib import Path


def load_env_file(project_root: Path = None):
    """
    Load environment variables from .env file if it exists.
    
    Args:
        project_root: Path to project root. If None, auto-detects from this file.
    
    Returns:
        True if .env file was found and loaded, False otherwise
    """
    if project_root is None:
        # Auto-detect project root (go up from experiments/utils/)
        project_root = Path(__file__).parent.parent.parent
    
    env_file = project_root / '.env'
    
    if not env_file.exists():
        return False
    
    # Try to use python-dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        return True
    except ImportError:
        # Fallback: manually parse .env file
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already set (env vars take precedence)
                    if key and not os.environ.get(key):
                        os.environ[key] = value
        return True


def ensure_openai_key():
    """
    Ensure OPENAI_API_KEY is set, loading from .env if needed.
    
    Returns:
        True if key is available, False otherwise
    """
    # Load .env if it exists
    load_env_file()
    
    # Check if key is set
    return bool(os.environ.get("OPENAI_API_KEY"))

