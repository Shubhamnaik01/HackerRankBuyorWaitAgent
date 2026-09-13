from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


EXCLUDED_DIRECTORIES = frozenset({
    ".cache", ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
})
EXCLUDED_NAMES = frozenset({
    ".env", "credentials.json", "secrets.json", "Thumbs.db", ".DS_Store",
})
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".tmp", ".bak", ".log")
SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
REQUIRED_PACKAGE_PATHS = frozenset({
    "README.md", "main.py", "requirements.txt", "evaluation/usage_report.md",
})


def is_excluded(relative: Path) -> bool:
    has_cache_directory = any(
        part in EXCLUDED_DIRECTORIES
        or part.lower() == "cache"
        or part.lower().endswith(("-cache", "_cache"))
        for part in relative.parts[:-1]
    )
    return (
        has_cache_directory
        or relative.name in EXCLUDED_DIRECTORIES
        or relative.name in EXCLUDED_NAMES
        or relative.name.startswith(".env.")
        or relative.name.endswith(EXCLUDED_SUFFIXES)
        or relative.name in {"output.csv", "code.zip"}
        or "debug" in relative.name.lower()
    )


def package_manifest(code_root: Path) -> tuple[Path, ...]:
    root = code_root.resolve()
    return tuple(
        path for path in sorted(root.rglob("*"))
        if path.is_file() and not is_excluded(path.relative_to(root))
    )


def validate_package_manifest(code_root: Path, files: tuple[Path, ...] | None = None) -> tuple[str, ...]:
    root = code_root.resolve()
    selected = files if files is not None else package_manifest(root)
    errors: list[str] = []
    relative_paths: set[str] = set()
    for path in selected:
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            errors.append(f"package path escapes code root: {path}")
            continue
        relative_text = relative.as_posix()
        relative_paths.add(relative_text)
        if is_excluded(relative):
            errors.append(f"excluded artifact selected for package: {relative_text}")
            continue
        try:
            content = path.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read package file {relative_text}: {exc}")
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                errors.append(f"possible secret in package file: {relative_text}")
                break
    for required in sorted(REQUIRED_PACKAGE_PATHS - relative_paths):
        errors.append(f"required package file missing: {required}")
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the safe code.zip file manifest")
    parser.add_argument(
        "--code-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    files = package_manifest(args.code_root)
    errors = validate_package_manifest(args.code_root, files)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Safe package manifest: {len(files)} files; caches, secrets, bytecode, and temporary artifacts excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
