"""Contract tests for status and list CLI commands."""

import pytest
import subprocess

@pytest.mark.contract
def test_status_command_exists():
    result = subprocess.run(["repo-kgraph", "status", "--help"], capture_output=True, text=True)
    assert result.returncode != 127
    assert "status" in result.stdout.lower()

@pytest.mark.contract
def test_list_command_exists():
    result = subprocess.run(["repo-kgraph", "list", "--help"], capture_output=True, text=True)
    assert result.returncode != 127
    assert "list" in result.stdout.lower() or "repositories" in result.stdout.lower()