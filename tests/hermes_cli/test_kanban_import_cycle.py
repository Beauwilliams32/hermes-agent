"""Managed Kanban workers import dispatcher modules before the board facade."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["kanban_db_dispatch", "kanban_db_guard"])
def test_kanban_modules_import_in_fresh_worker_process(module):
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", f"from hermes_cli import {module}; print('ready')"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ready"
