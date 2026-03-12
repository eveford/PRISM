from __future__ import annotations

import argparse
from pathlib import Path


FORBIDDEN_EXTENSIONS = {
    ".csv",
    ".h5ad",
    ".ipynb",
    ".joblib",
    ".npz",
    ".pkl",
    ".pt",
}
TEXT_EXTENSIONS = {
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
}
IGNORED_PATHS = {
    Path("scripts") / "audit_public_repo.py",
}
FORBIDDEN_TEXT_PATTERNS = [
    "/" + "data2404" + "/",
    "/" + "home" + "/",
    "C:" + "\\\\",
    "cuda:" + "7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the public repo tree for private files and paths.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def should_skip(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if relative in IGNORED_PATHS:
        return True
    return any(part in IGNORED_DIRS for part in relative.parts)


def iter_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not should_skip(path, root)
    ]


def audit_extensions(files: list[Path], root: Path) -> list[str]:
    findings: list[str] = []
    for path in files:
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            findings.append(f"Forbidden file extension: {path.relative_to(root)}")
    return findings


def audit_text(files: list[Path], root: Path) -> list[str]:
    findings: list[str] = []
    for path in files:
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern in text:
                findings.append(f"Forbidden text pattern '{pattern}' found in {path.relative_to(root)}")
    return findings


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    files = iter_files(root)
    findings = []
    findings.extend(audit_extensions(files, root))
    findings.extend(audit_text(files, root))

    if findings:
        for finding in findings:
            print(f"[FAIL] {finding}")
        raise SystemExit(1)

    print("Audit passed.")


if __name__ == "__main__":
    main()
