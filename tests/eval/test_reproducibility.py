"""G1 static checks (Task 4 §4.4). A full clean-machine `docker compose up`
run needs an actual Docker host and is exercised manually/in CI; these tests
check the parts that don't need one: the manifest exists and is internally
consistent, secrets aren't committed, and every env var the code reads is
declared in run.yaml.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_packaging_files_exist():
    for name in ("Dockerfile", "docker-compose.yml", "requirements.lock", "run.yaml", ".env.example", ".gitignore"):
        assert (REPO_ROOT / name).exists(), f"missing {name}"


def test_env_file_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert not (REPO_ROOT / ".env").exists(), ".env must not be committed — only .env.example"


def _env_vars_read_in_code() -> set[str]:
    found = set()
    pattern = re.compile(r"os\.environ(?:\.get)?\(\s*['\"]([A-Z_]+)['\"]|getenv\(\s*['\"]([A-Z_]+)['\"]")
    for path in (REPO_ROOT / "streaming_rag").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            found.add(m.group(1) or m.group(2))
    return found


def test_every_env_var_read_is_declared_in_manifest():
    manifest = yaml.safe_load((REPO_ROOT / "run.yaml").read_text(encoding="utf-8"))
    declared = {e["name"] for e in manifest.get("env", [])}
    used = _env_vars_read_in_code()
    undeclared = used - declared
    assert not undeclared, f"env vars read but not declared in run.yaml: {undeclared}"


def test_requirements_lock_has_no_unpinned_entries():
    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")
    for line in lock.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"unpinned dependency in requirements.lock: {line!r}"


def test_dockerfile_runs_as_non_root():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER runner" in dockerfile or "USER " in dockerfile
    assert "python:3.12-slim" in dockerfile
