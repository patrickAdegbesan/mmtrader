"""Versioned model registry (spec rule 8: every retrain is versioned,
rollback supported, no silent updates).

Layout on disk:
    models/<name>/v0001/model.pt
    models/<name>/v0001/meta.json
    models/<name>/LATEST            <- text file containing "v0001"

save() always creates a NEW version directory — existing versions are
immutable. activate() moves the LATEST pointer (that's also how you roll
back). Nothing ever deletes or overwrites a saved version.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch


class ModelRegistry:
    def __init__(self, root: Path, name: str):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)

    def _latest_file(self) -> Path:
        return self.dir / "LATEST"

    def list_versions(self) -> list[str]:
        return sorted(p.name for p in self.dir.iterdir() if p.is_dir() and p.name.startswith("v"))

    def latest_version(self) -> str | None:
        f = self._latest_file()
        if f.exists():
            version = f.read_text().strip()
            if (self.dir / version).is_dir():
                return version
        versions = self.list_versions()
        return versions[-1] if versions else None

    def save(self, state: dict, metadata: dict, activate: bool = True) -> str:
        versions = self.list_versions()
        next_num = int(versions[-1][1:]) + 1 if versions else 1
        version = f"v{next_num:04d}"
        vdir = self.dir / version
        vdir.mkdir()
        torch.save(state, vdir / "model.pt")
        meta = {
            "version": version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        if activate:
            self.activate(version)
        return version

    def activate(self, version: str) -> None:
        """Point LATEST at `version`. Rolling back = activating an older one."""
        if not (self.dir / version).is_dir():
            raise FileNotFoundError(f"No such version: {version}")
        self._latest_file().write_text(version)

    def load(self, version: str | None = None) -> tuple[dict, dict]:
        version = version or self.latest_version()
        if version is None:
            raise FileNotFoundError(f"Registry {self.dir} has no saved versions")
        vdir = self.dir / version
        state = torch.load(vdir / "model.pt", weights_only=False)
        meta = json.loads((vdir / "meta.json").read_text())
        return state, meta
