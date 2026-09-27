"""Pytest configuration — add project root to sys.path."""

import sys
from pathlib import Path

# Add project root to sys.path so tests can import core/ and components/
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
