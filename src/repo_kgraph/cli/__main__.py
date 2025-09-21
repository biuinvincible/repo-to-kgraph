"""
Main CLI entry point for the repository knowledge graph system.

Provides the primary command-line interface with subcommands for
repository parsing, querying, and system management.
"""

import sys
import asyncio
from typing import Optional, List, Dict, Any
import click

from repo_kgraph.cli.base import (
    console, handle_cli_errors, async_command, setup_cli_context,
    set_cli_context, get_cli_context, add_common_options,
    print_success, print_error, print_info
)


@click.group()
@click.version_option(version="1.0.0", prog_name="repo-kgraph")
@click.pass_context
def cli(ctx):
    """
    Repository Knowledge Graph System

    Transform code repositories into queryable knowledge graphs for AI coding agents.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)


@cli.command("parse-repo")
@click.argument("repository_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--incremental", is_flag=True, help="Perform incremental parsing (only changed files)")
@click.option("--languages", multiple=True, help="Programming languages to include")
@click.option("--exclude", multiple=True, help="File patterns to exclude")
@click.option("--repository-id", help="Custom repository identifier")
@click.option("--reset-repo", is_flag=True, help="Reset and recreate the database for this repository")
@add_common_options
@handle_cli_errors
@async_command
async def parse_repo(
    repository_path: str,
    incremental: bool,
    languages: tuple,
    exclude: tuple,
    repository_id: Optional[str],
    reset_repo: bool,
    config: Optional[str],
    env_file: Optional[str],
    verbose: bool
):
    """Parse a repository and build its knowledge graph."""
    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)

    try:
        await context.initialize()

        print_info(f"Parsing repository: {repository_path}")

        # Get repository manager
        repo_manager = context.get_service("repository_manager")

        # Reset repository if requested
        if reset_repo and repository_id:
            print_info(f"Resetting repository: {repository_id}")
            await repo_manager.delete_repository(repository_id)

        # Setup progress callback
        async def progress_callback(repo_id: str, progress: float, phase: str):
            print_info(f"Progress: {progress:.1f}% - {phase}")

        # Process repository
        repository, knowledge_graph = await repo_manager.process_repository(
            repository_path=repository_path,
            repository_id=repository_id,
            incremental=incremental,
            languages=list(languages) if languages else None,
            exclude_patterns=list(exclude) if exclude else None,
            progress_callback=progress_callback
        )

        print_success(f"Repository processed successfully!")
        print_info(f"Repository ID: {repository.id}")
        print_info(f"Entities: {repository.entity_count}")
        print_info(f"Relationships: {repository.relationship_count}")
        print_info(f"Files: {repository.file_count}")
        print_info(f"Processing time: {repository.indexing_time_ms / 1000:.2f}s")

    except Exception as e:
        print_error(f"Repository parsing failed: {e}")
        raise
    finally:
        await context.cleanup()


@cli.command("query")
@click.argument("task_description")
@click.option("--repository-id", help="Repository to search in")
@click.option("--max-results", type=int, default=20, help="Maximum number of results")
@click.option("--confidence", type=float, default=0.3, help="Minimum confidence threshold")
@click.option("--language", multiple=True, help="Programming language filter")
@click.option("--entity-type", multiple=True, help="Entity type filter")
@click.option("--file-filter", help="File path filter")
@click.option("--no-related", is_flag=True, help="Exclude related entities")
@click.option("--output-format", type=click.Choice(["table", "json", "detailed"]), default="table")
@add_common_options
@handle_cli_errors
@async_command
async def query_task(
    task_description: str,
    repository_id: Optional[str],
    max_results: int,
    confidence: float,
    language: tuple,
    entity_type: tuple,
    file_filter: Optional[str],
    no_related: bool,
    output_format: str,
    config: Optional[str],
    env_file: Optional[str],
    verbose: bool
):
    """Query for relevant code context based on task description."""
    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)

    try:
        await context.initialize()

        # Auto-detect repository if not provided
        if not repository_id:
            print_error("Repository ID is required. Use 'repo-kgraph list' to see available repositories.")
            return

        print_info(f"Querying: {task_description}")

        # Get query processor
        query_processor = context.get_service("query_processor")

        # Convert entity types
        from repo_kgraph.models.code_entity import EntityType
        entity_types = None
        if entity_type:
            entity_types = []
            for et in entity_type:
                try:
                    entity_types.append(EntityType(et.upper()))
                except ValueError:
                    print_error(f"Invalid entity type: {et}")
                    return

        # Process query
        result = await query_processor.process_query(
            repository_id=repository_id,
            task_description=task_description,
            max_results=max_results,
            confidence_threshold=confidence,
            language_filter=list(language) if language else None,
            entity_types=entity_types,
            file_path_filter=file_filter,
            include_related=not no_related
        )

        # Display results
        if output_format == "json":
            import json
            console.print_json(json.dumps(result, indent=2))
        elif output_format == "detailed":
            _display_detailed_results(result)
        else:
            _display_table_results(result)

        print_success(f"Found {result['total_results']} results in {result['processing_time_ms']}ms")

    except Exception as e:
        print_error(f"Query failed: {e}")
        raise
    finally:
        await context.cleanup()


@cli.command("serve")
@click.option("--host", default="localhost", help="Server host")
@click.option("--port", type=int, default=8000, help="Server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@add_common_options
@handle_cli_errors
def serve(
    host: str,
    port: int,
    reload: bool,
    config: Optional[str],
    env_file: Optional[str],
    verbose: bool
):
    """Start the API server."""
    import uvicorn
    from repo_kgraph.api.main import app

    print_info(f"Starting API server on {host}:{port}")

    if reload:
        print_info("Auto-reload enabled")

    # Set environment variables for the app
    import os
    if config:
        os.environ["CONFIG_FILE"] = config
    if env_file:
        os.environ["ENV_FILE"] = env_file
    if verbose:
        os.environ["VERBOSE"] = "true"

    uvicorn.run(
        "repo_kgraph.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="debug" if verbose else "info"
    )


@cli.command("update")
@click.argument("repository_id")
@click.argument("changed_files", nargs=-1)
@add_common_options
@handle_cli_errors
@async_command
async def update(
    repository_id: str,
    changed_files: tuple,
    config: Optional[str],
    env_file: Optional[str],
    verbose: bool
):
    """Update repository with changed files."""
    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)

    try:
        await context.initialize()

        print_info(f"Updating repository: {repository_id}")
        print_info(f"Changed files: {len(changed_files)}")

        # Get repository manager
        repo_manager = context.get_service("repository_manager")

        # Setup progress callback
        async def progress_callback(repo_id: str, progress: float, phase: str):
            print_info(f"Progress: {progress:.1f}% - {phase}")

        # Update repository
        success = await repo_manager.update_repository_incremental(
            repository_id=repository_id,
            changed_files=list(changed_files),
            progress_callback=progress_callback
        )

        if success:
            print_success("Repository updated successfully")
        else:
            print_error("Repository update failed")

    except Exception as e:
        print_error(f"Repository update failed: {e}")
        raise
    finally:
        await context.cleanup()


@cli.command("status")
@click.argument("repository_id", required=False)
@add_common_options
@handle_cli_errors
@async_command
async def status(
    repository_id: Optional[str],
    config: Optional[str],
    env_file: Optional[str],
    verbose: bool
):
    """Show repository status and statistics."""
    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)

    try:
        await context.initialize()

        if repository_id:
            # Show specific repository status
            repo_manager = context.get_service("repository_manager")
            stats = await repo_manager.get_repository_statistics(repository_id)

            if stats:
                _display_repository_stats(repository_id, stats)
            else:
                print_error(f"Repository not found: {repository_id}")
        else:
            # Show overall system status
            await _display_system_status(context)

    except Exception as e:
        print_error(f"Status check failed: {e}")
        raise
    finally:
        await context.cleanup()


@cli.command("list")
@add_common_options
@handle_cli_errors
@async_command
async def list_repos(
    config: Optional[str],
    env_file: Optional[str],
    verbose: bool
):
    """List all indexed repositories."""
    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)

    try:
        await context.initialize()

        # Get repository manager
        repo_manager = context.get_service("repository_manager")

        # List repositories
        repositories = await repo_manager.list_repositories()

        if not repositories:
            print_info("No repositories found")
            return

        # Display repositories in table format
        _display_repositories_table(repositories)

        print_success(f"Found {len(repositories)} repositories")

    except Exception as e:
        print_error(f"Repository listing failed: {e}")
        raise
    finally:
        await context.cleanup()


@cli.command("clear")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@add_common_options
@handle_cli_errors
@async_command
async def clear_all(
    confirm: bool,
    config: Optional[str],
    env_file: Optional[str],
    verbose: bool
):
    """Clear all parsed repositories and their data."""
    # Setup CLI context
    context = setup_cli_context(config, env_file, verbose)
    set_cli_context(context)

    try:
        await context.initialize()

        # Get repository manager
        repo_manager = context.get_service("repository_manager")

        # Get list of repositories first to show what will be deleted
        repositories = await repo_manager.list_repositories()

        if not repositories:
            print_info("No repositories found to clear")
            return

        # Show what will be deleted
        print_info(f"Found {len(repositories)} repositories to clear:")
        for repo in repositories:
            print_info(f"  - {repo.get('repository_name', 'Unknown')} ({repo.get('repository_id', 'Unknown ID')})")

        # Confirmation prompt unless --confirm flag is used
        if not confirm:
            console.print("\n[yellow]WARNING: This will permanently delete all repository data![/yellow]")
            console.print("[yellow]This includes all entities, relationships, and embeddings.[/yellow]")

            response = console.input("\n[bold]Are you sure you want to continue? (yes/no): [/bold]")
            if response.lower() not in ['yes', 'y']:
                print_info("Operation cancelled")
                return

        print_info("Clearing all repository data...")

        # Clear all repositories
        success = await repo_manager.clear_all_repositories()

        if success:
            print_success("All repository data cleared successfully")
        else:
            print_error("Some repositories could not be cleared (check logs for details)")

    except Exception as e:
        print_error(f"Clear operation failed: {e}")
        raise
    finally:
        await context.cleanup()


def _display_table_results(result: dict):
    """Display query results in table format."""
    from repo_kgraph.cli.base import print_table

    if not result.get("results"):
        print_info("No results found")
        return

    headers = ["Name", "Type", "File", "Lines", "Score", "Reason"]
    rows = []

    for res in result["results"]:
        rows.append([
            res["name"],
            res["entity_type"],
            res["file_path"],
            f"{res['start_line']}-{res['end_line']}",
            f"{res['relevance_score']:.3f}",
            res["match_reason"]
        ])

    print_table(rows, headers, "Query Results")


def _display_detailed_results(result: dict):
    """Display query results in detailed format."""
    if not result.get("results"):
        print_info("No results found")
        return

    for i, res in enumerate(result["results"], 1):
        console.print(f"\n[bold]Result {i}:[/bold] {res['name']}")
        console.print(f"[dim]Type:[/dim] {res['entity_type']}")
        console.print(f"[dim]File:[/dim] {res['file_path']}:{res['start_line']}-{res['end_line']}")
        console.print(f"[dim]Score:[/dim] {res['relevance_score']:.3f}")
        console.print(f"[dim]Reason:[/dim] {res['match_reason']}")

        if res.get("context"):
            console.print(f"[dim]Context:[/dim]\n{res['context']}")


def _display_repository_stats(repository_id: str, stats: dict):
    """Display repository statistics."""
    from repo_kgraph.cli.base import print_panel

    content = f"""Repository ID: {repository_id}
Total Entities: {stats.get('total_entities', 0)}
Total Relationships: {stats.get('total_relationships', 0)}
File Count: {stats.get('file_count', 0)}

Language Distribution:
"""

    lang_dist = stats.get('language_distribution', {})
    for lang, count in lang_dist.items():
        content += f"  {lang}: {count}\n"

    content += f"""
Embedding Info:
  Provider: {stats.get('embedding_info', {}).get('embedding_provider', 'N/A')}
  Model: {stats.get('embedding_info', {}).get('embedding_model', 'N/A')}
  Dimension: {stats.get('embedding_info', {}).get('embedding_dimension', 'N/A')}
"""

    print_panel(content, title="Repository Statistics", style="green")


def _display_repositories_table(repositories: List[Dict[str, Any]]):
    """Display repositories in table format."""
    from repo_kgraph.cli.base import print_table
    from datetime import datetime

    if not repositories:
        print_info("No repositories found")
        return

    headers = ["Repository ID", "Name", "Path", "Entities", "Relationships", "Status", "Updated"]
    rows = []

    for repo in repositories:
        # Format the updated_at date
        updated_at = repo.get("updated_at")
        if updated_at:
            if isinstance(updated_at, str):
                try:
                    updated_dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    updated_str = updated_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    updated_str = updated_at[:16] if len(updated_at) > 16 else updated_at
            else:
                updated_str = str(updated_at)[:16]
        else:
            updated_str = "Unknown"

        # Truncate long paths for better display
        path = repo.get("repository_path", "")
        if len(path) > 40:
            path = "..." + path[-37:]

        rows.append([
            repo.get("repository_id", ""),
            repo.get("repository_name", ""),
            path,
            str(repo.get("entity_count", 0)),
            str(repo.get("relationship_count", 0)),
            repo.get("status", "Unknown"),
            updated_str
        ])

    print_table(rows, headers, "Indexed Repositories")


async def _display_system_status(context):
    """Display overall system status."""
    from repo_kgraph.cli.base import print_panel

    # Check database health
    health = await context.database_manager.health_check()

    content = f"""System Status: {"Healthy" if health['healthy'] else "Degraded"}

Database Components:
  Neo4j: {"✓" if health['components']['neo4j']['healthy'] else "✗"} {health['components']['neo4j']['status']}
  ChromaDB: {"✓" if health['components']['chroma']['healthy'] else "✗"} {health['components']['chroma']['status']}

Configuration:
  Data Directory: {context.config.data_dir}
  Log Directory: {context.config.log_dir}
  Embedding Provider: {context.config.embedding.embedding_provider}
  Model: {context.config.embedding.model_name}
"""

    style = "green" if health['healthy'] else "red"
    print_panel(content, title="System Status", style=style)


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()