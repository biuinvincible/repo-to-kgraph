"""Contract tests for serve CLI command."""

import pytest
import subprocess

@pytest.mark.contract
def test_serve_command_exists():
    result = subprocess.run(["repo-kgraph", "serve", "--help"], capture_output=True, text=True)
    assert result.returncode != 127
    assert "serve" in result.stdout.lower()

@pytest.mark.contract
def test_serve_port_option():
    # Just test that the option is recognized
    result = subprocess.run(["repo-kgraph", "serve", "--port", "8001", "--help"], capture_output=True, text=True)
    assert result.returncode != 2  # Not invalid option