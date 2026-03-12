from __future__ import annotations

from pathlib import Path


def require_relative_path(path_value: str | Path, *, label: str) -> Path:
    """Reject absolute paths inside tracked public configs."""
    path = Path(path_value)
    if path.is_absolute():
        raise ValueError(f"{label} must be a relative path inside the public repository.")
    return path


def resolve_relative_path(base_dir: Path, relative_path: str | Path) -> Path:
    return (base_dir / Path(relative_path)).resolve()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
