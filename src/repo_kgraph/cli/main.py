"""
Main CLI entry point for repo-kgraph.

Provides commands for parsing repositories and querying the knowledge graph.
"""

import json
import uuid
import os
from datetime import datetime
from pathlib import Path
import click

from repo_kgraph.cli.base import (
    add_common_options, handle_cli_errors, async_command,
    setup_cli_context, set_cli_context, validate_repository_path,
    print_success, print_info, print_error, format_file_size,
    ProgressReporter
)
from repo_kgraph.models.repository import Repository


@click.group()
@click.version_option(version="1.0.0")
@click.option("--config", help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
def main(config, verbose, quiet):
    """Repository Knowledge Graph CLI tool."""
    pass


@main.command()
@click.help_option("--help", "-h")
@click.option("--output", "-o", help="Output directory for graph files")
@click.option("--incremental", "-i", is_flag=True, help="Perform incremental update")
@click.option("--languages", "-l", help="Comma-separated list of languages to parse")
@click.option("--exclude", "-e", multiple=True, help="Exclude patterns (can be used multiple times)")
@click.option("--format", "-f", default="json", help="Output format (json, graphml, etc.)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.argument("repository_path", type=click.Path(exists=True), required=False)
@add_common_options
@handle_cli_errors
@async_command
async def parse_repo(repository_path, output, incremental, languages, exclude, format, verbose, config, env_file):
    """Parse repository and build knowledge graph."""

    if not repository_path:
        click.echo("Error: Repository path is required", err=True)
        raise click.Abort()

    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)
    await context.initialize()

    try:
        # Parse languages if provided
        language_list = []
        if languages:
            language_list = [lang.strip() for lang in languages.split(",")]

        # Validate repository path
        repo_path = validate_repository_path(repository_path)

        # Generate repository ID
        repo_id = str(uuid.uuid4())

        # Get repository manager service
        repo_manager = context.get_service("repository_manager")

        # For JSON format, output initial response then process
        if format.lower() == "json":
            response = {
                "repository_id": repo_id,
                "status": "processing",
                "message": f"Started parsing repository: {repository_path}"
            }

            if verbose:
                response["output_directory"] = output
                response["incremental"] = incremental
                response["languages"] = language_list
                response["exclude_patterns"] = list(exclude)
                response["started_at"] = datetime.utcnow().isoformat() + "Z"

            click.echo(json.dumps(response, indent=2))
        else:
            print_info(f"🔍 Parsing repository: {repository_path}")

            if language_list:
                print_info(f"🔤 Languages: {', '.join(language_list)}")
            if exclude:
                print_info(f"🚫 Exclude patterns: {', '.join(exclude)}")

        # Parse the repository
        with ProgressReporter("Parsing repository") as progress:
            progress.update(10, "Initializing parser...")

            repository, knowledge_graph = await repo_manager.process_repository(
                repository_path=str(repo_path),
                repository_id=repo_id,
                incremental=incremental,
                languages=language_list,
                exclude_patterns=list(exclude) if exclude else None
            )

            progress.update(100, "Parsing complete!")

        result = repository

        # Output results
        if format.lower() == "json":
            final_response = {
                "repository_id": repo_id,
                "status": "completed",
                "message": f"Successfully parsed repository: {repository_path}",
                "entity_count": result.entity_count if hasattr(result, 'entity_count') else 0,
                "relationship_count": result.relationship_count if hasattr(result, 'relationship_count') else 0,
                "completed_at": datetime.utcnow().isoformat() + "Z"
            }
            click.echo(json.dumps(final_response, indent=2))
        else:
            print_success("✅ Repository parsing completed!")
            if hasattr(result, 'entity_count'):
                print_info(f"📝 Entities extracted: {result.entity_count:,}")
            if hasattr(result, 'relationship_count'):
                print_info(f"🔗 Relationships found: {result.relationship_count:,}")

    finally:
        # Cleanup context
        await context.cleanup()


@main.command()
@click.help_option("--help", "-h")
@click.option("--format", "-f", default="json", help="Output format (json, text, etc.)")
@click.option("--repository", "-r", help="Repository ID to query (defaults to current)")
@click.option("--max-results", "-n", default=20, help="Maximum number of results to return")
@click.option("--threshold", "-t", default=0.3, help="Confidence threshold for results")
@click.argument("task_description", type=str, required=False)
@add_common_options
@handle_cli_errors
@async_command
async def query_task(task_description, format, repository, max_results, threshold, config, env_file, verbose):
    """Retrieve relevant context for coding task."""
    if not task_description:
        click.echo("Error: Task description is required", err=True)
        raise click.Abort()

    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)
    await context.initialize()

    try:
        # Get query processor service
        query_processor = context.get_service("query_processor")

        # Process query
        result = await query_processor.process_query(
            task_description,
            repository_id=repository,
            max_results=max_results,
            confidence_threshold=threshold
        )

        # Output results
        if format.lower() == "json":
            click.echo(json.dumps(result.model_dump(), indent=2))
        else:
            print_success(f"Found {len(result.context_items)} relevant items for: {task_description}")
            for i, item in enumerate(result.context_items):
                print_info(f"{i+1}. {item.name} ({item.entity_type}) - {item.confidence:.2f}")

    finally:
        await context.cleanup()


if __name__ == "__main__":
    main()