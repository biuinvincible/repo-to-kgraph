"""Contract tests for update CLI command."""

import pytest
import subprocess

@pytest.mark.contract
def test_update_command_exists():
    result = subprocess.run(["repo-kgraph", "update", "--help"], capture_output=True, text=True)
    assert result.returncode != 127
    assert "update" in result.stdout.lower()

@pytest.mark.contract
def test_update_incremental_flag():
    result = subprocess.run(["repo-kgraph", "update", "--force", "--help"], capture_output=True, text=True)
    assert result.returncode != 2