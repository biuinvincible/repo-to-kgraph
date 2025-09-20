"""Contract tests for query-task CLI command."""

import pytest
import subprocess
import json

@pytest.mark.contract
def test_query_task_command_exists():
    result = subprocess.run(
        ["repo-kgraph", "query-task", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode != 127
    assert "query-task" in result.stdout or "query-task" in result.stderr

@pytest.mark.contract
def test_query_task_requires_description():
    result = subprocess.run(
        ["repo-kgraph", "query-task"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "description" in result.stderr.lower() or "argument" in result.stderr.lower()

@pytest.mark.contract
def test_query_task_json_output():
    result = subprocess.run([
        "repo-kgraph", "query-task", 
        "--format", "json",
        "test task description"
    ], capture_output=True, text=True)
    
    # Should accept format option
    assert result.returncode != 2