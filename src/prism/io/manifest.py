from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from prism.paths import require_relative_path


@dataclass(frozen=True)
class DataManifest:
    manifest_path: Path
    data_root: Path
    files: dict[str, Path]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataManifest":
        manifest_path = Path(path).resolve()
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a mapping in {manifest_path}.")

        data_root = require_relative_path(payload.get("data_root", "."), label="data_root")
        raw_files = payload.get("files", {})
        if not isinstance(raw_files, dict):
            raise ValueError("The 'files' section must be a mapping of logical names to relative paths.")

        files: dict[str, Path] = {}
        for logical_name, relative_path in raw_files.items():
            files[str(logical_name)] = require_relative_path(
                relative_path,
                label=f"files.{logical_name}",
            )

        return cls(manifest_path=manifest_path, data_root=data_root, files=files)

    def resolve(self, logical_name: str) -> Path:
        try:
            relative_path = self.files[logical_name]
        except KeyError as exc:
            raise KeyError(f"Logical file '{logical_name}' is not defined in {self.manifest_path}.") from exc

        env_root = os.getenv("PRISM_DATA_DIR")
        if env_root:
            return (Path(env_root).expanduser().resolve() / relative_path).resolve()
        return (self.manifest_path.parent / self.data_root / relative_path).resolve()
