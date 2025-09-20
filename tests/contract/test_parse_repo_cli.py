"""
Contract tests for parse-repo CLI command.

These tests define the expected behavior of the CLI and MUST FAIL initially.
"""

import pytest
import subprocess
import json
from pathlib import Path


@pytest.mark.contract
def test_parse_repo_command_exists():
    """Test that parse-repo command exists and is accessible."""
    # This will fail until we implement the CLI
    result = subprocess.run(
        ["repo-kgraph", "parse-repo", "--help"],
        capture_output=True,
        text=True
    )

    # Should not return command not found error
    assert result.returncode != 127, "repo-kgraph command should be available"
    assert "parse-repo" in result.stdout or "parse-repo" in result.stderr


@pytest.mark.contract
def test_parse_repo_help_output():
    """Test that parse-repo shows proper help information."""
    result = subprocess.run(
        ["repo-kgraph", "parse-repo", "--help"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    help_text = result.stdout.lower()

    # Should describe main functionality
    assert "parse" in help_text
    assert "repository" in help_text

    # Should show important options
    assert "--output" in help_text or "-o" in help_text
    assert "--incremental" in help_text or "-i" in help_text


@pytest.mark.contract
def test_parse_repo_missing_path():
    """Test error handling when repository path is not provided."""
    result = subprocess.run(
        ["repo-kgraph", "parse-repo"],
        capture_output=True,
        text=True
    )

    # Should return error code for missing argument
    assert result.returncode != 0

    error_output = result.stderr.lower()
    assert "path" in error_output or "argument" in error_output


@pytest.mark.contract
def test_parse_repo_json_output(temp_dir):
    """Test that parse-repo supports JSON output format."""
    result = subprocess.run(
        ["repo-kgraph", "parse-repo", "--format", "json", str(temp_dir)],
        capture_output=True,
        text=True
    )

    # Command should exist and attempt to parse
    assert result.returncode != 127  # Not "command not found"

    # If successful, should output valid JSON
    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            assert "repository_id" in data or "status" in data
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")


@pytest.mark.contract
def test_parse_repo_incremental_flag(temp_dir):
    """Test incremental parsing flag."""
    result = subprocess.run(
        ["repo-kgraph", "parse-repo", "--incremental", str(temp_dir)],
        capture_output=True,
        text=True
    )

    # Should accept incremental flag without error
    assert result.returncode != 2  # Not "invalid option" error


@pytest.mark.contract
def test_parse_repo_output_directory(temp_dir):
    """Test specifying output directory."""
    output_dir = temp_dir / "output"

    result = subprocess.run([
        "repo-kgraph", "parse-repo",
        "--output", str(output_dir),
        str(temp_dir)
    ], capture_output=True, text=True)

    # Should accept output option
    assert result.returncode != 2  # Not "invalid option" error


@pytest.mark.contract
def test_parse_repo_language_filter(temp_dir):
    """Test language filtering option."""
    result = subprocess.run([
        "repo-kgraph", "parse-repo",
        "--languages", "python,javascript",
        str(temp_dir)
    ], capture_output=True, text=True)

    # Should accept languages option
    assert result.returncode != 2  # Not "invalid option" error


@pytest.mark.contract
def test_parse_repo_exclude_patterns(temp_dir):
    """Test exclude patterns option."""
    result = subprocess.run([
        "repo-kgraph", "parse-repo",
        "--exclude", "*.test.py",
        "--exclude", "__pycache__/*",
        str(temp_dir)
    ], capture_output=True, text=True)

    # Should accept exclude options
    assert result.returncode != 2  # Not "invalid option" error


@pytest.mark.contract
def test_parse_repo_verbose_output(temp_dir):
    """Test verbose output option."""
    result = subprocess.run([
        "repo-kgraph", "parse-repo",
        "--verbose",
        str(temp_dir)
    ], capture_output=True, text=True)

    # Should accept verbose flag
    assert result.returncode != 2  # Not "invalid option" error