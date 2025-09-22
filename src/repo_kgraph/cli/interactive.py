"""
Interactive CLI shell for the repository knowledge graph system.

Provides a Claude Code-like interface for faster interaction without
repeatedly typing commands.
"""

import asyncio
import sys
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn
import readline

from repo_kgraph.cli.base import (
    console, CLIContext, get_cli_context, set_cli_context,
    print_success, print_error, print_info, print_warning,
    ProgressReporter, validate_repository_path
)


class InteractiveCLI:
    """Interactive CLI shell with Claude Code-like interface."""

    def __init__(self):
        self.context: Optional[CLIContext] = None
        self.console = Console()
        self.current_repo_id: Optional[str] = None
        self.repositories: Dict[str, Dict[str, Any]] = {}
        self.running = True
        self.history = []

        # Command mappings
        self.commands = {
            'help': self.show_help,
            'h': self.show_help,
            '?': self.show_help,
            'parse': self.parse_repository,
            'p': self.parse_repository,
            'query': self.query_context,
            'q': self.query_context,
            'list': self.list_repositories,
            'ls': self.list_repositories,
            'status': self.show_status,
            'st': self.show_status,
            'use': self.switch_repository,
            'switch': self.switch_repository,
            'clear': self.clear_screen,
            'cls': self.clear_screen,
            'config': self.show_config,
            'exit': self.exit_shell,
            'quit': self.exit_shell,
            'q!': self.exit_shell,
        }

        # Setup readline for command history
        self._setup_readline()

    def _setup_readline(self):
        """Setup readline for command history and completion."""
        try:
            # Enable history
            readline.set_startup_hook(None)

            # Setup completion
            def completer(text, state):
                options = [cmd for cmd in self.commands.keys() if cmd.startswith(text)]
                if state < len(options):
                    return options[state]
                return None

            readline.set_completer(completer)
            readline.parse_and_bind("tab: complete")

            # Load history
            history_file = os.path.expanduser("~/.kgraph_history")
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
        except ImportError:
            # readline not available on some systems
            pass

    def _save_history(self):
        """Save command history."""
        try:
            history_file = os.path.expanduser("~/.kgraph_history")
            readline.set_history_length(1000)
            readline.write_history_file(history_file)
        except (ImportError, PermissionError):
            pass

    async def initialize(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        """Initialize the CLI context."""
        try:
            self.context = CLIContext(config_file, env_file)
            await self.context.initialize()
            set_cli_context(self.context)

            # Load existing repositories
            await self._load_repositories()

            print_success("Interactive CLI initialized successfully!")

        except Exception as e:
            print_error(f"Failed to initialize CLI context: {e}")
            raise

    async def _load_repositories(self):
        """Load existing repositories from the database."""
        try:
            repo_manager = self.context.get_service("repository_manager")
            # This would need to be implemented in the repository manager
            # For now, we'll simulate it
            self.repositories = {}
        except Exception as e:
            print_warning(f"Could not load repositories: {e}")

    def show_welcome(self):
        """Display welcome message and help."""
        welcome_text = Text()
        welcome_text.append("🚀 Repository Knowledge Graph - Interactive CLI\n", style="bold blue")
        welcome_text.append("Type 'help' or '?' for available commands\n", style="dim")
        welcome_text.append("Type 'exit' or 'quit' to leave\n", style="dim")

        if self.current_repo_id:
            welcome_text.append(f"\n📂 Current repository: {self.current_repo_id}", style="green")

        panel = Panel(welcome_text, title="Welcome", border_style="blue")
        self.console.print(panel)

    def show_help(self, args: List[str] = None):
        """Show available commands."""
        table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan", min_width=15)
        table.add_column("Shortcut", style="yellow", min_width=8)
        table.add_column("Description", style="white")

        commands_help = [
            ("help", "h, ?", "Show this help message"),
            ("parse <path>", "p", "Parse a repository and add to knowledge graph"),
            ("query <text>", "q", "Query for relevant code context"),
            ("list", "ls", "List all indexed repositories"),
            ("status [repo]", "st", "Show repository status (current if no repo specified)"),
            ("use <repo-id>", "switch", "Switch to a different repository"),
            ("config", "", "Show current configuration"),
            ("clear", "cls", "Clear the screen"),
            ("exit", "quit, q!", "Exit the interactive shell"),
        ]

        for cmd, shortcut, desc in commands_help:
            table.add_row(cmd, shortcut, desc)

        self.console.print(table)

        # Show current context info
        if self.current_repo_id:
            self.console.print(f"\n💡 Current repository: [green]{self.current_repo_id}[/green]")
        else:
            self.console.print("\n💡 No repository selected. Use [cyan]list[/cyan] to see available repositories or [cyan]parse <path>[/cyan] to add one.")

    async def parse_repository(self, args: List[str]):
        """Parse a repository."""
        if not args:
            print_error("Usage: parse <repository-path>")
            print_info("Example: parse /path/to/my/project")
            return

        repo_path = args[0]

        try:
            # Validate path
            validated_path = validate_repository_path(repo_path)

            print_info(f"Parsing repository: {validated_path}")

            # Get repository manager
            repo_manager = self.context.get_service("repository_manager")

            # Show progress
            with ProgressReporter("Parsing repository") as progress:
                progress.update(10, "Initializing...")

                # Parse repository
                repository, knowledge_graph = await repo_manager.process_repository(
                    repository_path=validated_path,
                    reset_repo=False  # Default to incremental
                )

                progress.update(100, "Completed!")

            print_success(f"Repository parsed successfully!")
            print_info(f"Repository ID: {repository.id}")
            print_info(f"Entities: {len(knowledge_graph.entities) if knowledge_graph else 'N/A'}")

            # Auto-switch to this repository
            self.current_repo_id = repository.id
            self.repositories[repository.id] = {
                'name': repository.name,
                'path': str(validated_path),
                'entities': len(knowledge_graph.entities) if knowledge_graph else 0
            }

            print_success(f"Switched to repository: {repository.id}")

        except Exception as e:
            print_error(f"Failed to parse repository: {e}")

    async def query_context(self, args: List[str]):
        """Query for relevant code context."""
        if not args:
            print_error("Usage: query <search-text>")
            print_info("Example: query user authentication")
            return

        if not self.current_repo_id:
            print_error("No repository selected. Use 'list' to see available repositories or 'use <repo-id>' to select one.")
            return

        query_text = " ".join(args)

        try:
            print_info(f"🔍 Searching: {query_text}")
            print_info(f"📂 Repository: {self.current_repo_id}")

            # Get query processor
            query_processor = self.context.get_service("query_processor")

            # Execute query
            with self.console.status("[bold green]Searching...") as status:
                results = await query_processor.query(
                    task_description=query_text,
                    repository_id=self.current_repo_id,
                    max_results=10,
                    confidence_threshold=0.3
                )

            if not results or not results.results:
                print_warning("No results found. Try:")
                print_info("- Lowering confidence with broader search terms")
                print_info("- Checking if the repository was fully indexed")
                return

            # Display results
            self._display_query_results(query_text, results)

        except Exception as e:
            print_error(f"Query failed: {e}")

    def _display_query_results(self, query: str, results):
        """Display query results in a nice format."""
        table = Table(title=f"🔍 Results for: {query}", show_header=True, header_style="bold magenta")
        table.add_column("Score", style="green", width=8)
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Name", style="yellow", width=20)
        table.add_column("File", style="blue", width=30)
        table.add_column("Description", style="white")

        for i, result in enumerate(results.results[:10], 1):
            table.add_row(
                f"{result.confidence:.2f}",
                result.entity.entity_type.value if hasattr(result.entity, 'entity_type') else "N/A",
                result.entity.name[:18] + "..." if len(result.entity.name) > 18 else result.entity.name,
                f"{result.entity.file_path}:{result.entity.line_start}" if hasattr(result.entity, 'line_start') else result.entity.file_path,
                (result.entity.description[:40] + "...") if len(result.entity.description or "") > 40 else (result.entity.description or "")
            )

        self.console.print(table)
        print_info(f"Found {len(results.results)} results")

    async def list_repositories(self, args: List[str] = None):
        """List all indexed repositories."""
        try:
            # This would call the actual repository listing service
            # For now, show what we have in memory
            if not self.repositories:
                print_warning("No repositories indexed yet.")
                print_info("Use 'parse <path>' to index a repository.")
                return

            table = Table(title="📚 Indexed Repositories", show_header=True, header_style="bold magenta")
            table.add_column("Repository ID", style="cyan")
            table.add_column("Name", style="yellow")
            table.add_column("Path", style="blue")
            table.add_column("Entities", style="green")
            table.add_column("Status", style="white")

            for repo_id, info in self.repositories.items():
                status = "🟢 ACTIVE" if repo_id == self.current_repo_id else "⚪ Available"
                table.add_row(
                    repo_id[:12] + "..." if len(repo_id) > 15 else repo_id,
                    info.get('name', 'Unknown'),
                    info.get('path', 'Unknown')[:30] + "..." if len(info.get('path', '')) > 30 else info.get('path', 'Unknown'),
                    str(info.get('entities', 0)),
                    status
                )

            self.console.print(table)

            if self.current_repo_id:
                print_info(f"Current repository: {self.current_repo_id}")
            else:
                print_info("Use 'use <repo-id>' to select a repository")

        except Exception as e:
            print_error(f"Failed to list repositories: {e}")

    async def show_status(self, args: List[str] = None):
        """Show repository status."""
        repo_id = args[0] if args else self.current_repo_id

        if not repo_id:
            print_error("No repository specified and no current repository selected.")
            print_info("Usage: status <repo-id> or select a repository with 'use <repo-id>'")
            return

        try:
            print_info(f"Repository Status: {repo_id}")

            # This would call the actual status service
            # For now, show basic info
            if repo_id in self.repositories:
                info = self.repositories[repo_id]
                self.console.print(f"📂 Name: {info.get('name', 'Unknown')}")
                self.console.print(f"📁 Path: {info.get('path', 'Unknown')}")
                self.console.print(f"📊 Entities: {info.get('entities', 0)}")
                self.console.print(f"🎯 Status: {'Active' if repo_id == self.current_repo_id else 'Available'}")
            else:
                print_warning(f"Repository {repo_id} not found in memory.")
                print_info("Use 'list' to see available repositories.")

        except Exception as e:
            print_error(f"Failed to get status: {e}")

    def switch_repository(self, args: List[str]):
        """Switch to a different repository."""
        if not args:
            print_error("Usage: use <repository-id>")
            print_info("Use 'list' to see available repositories.")
            return

        repo_id = args[0]

        if repo_id in self.repositories:
            self.current_repo_id = repo_id
            print_success(f"Switched to repository: {repo_id}")
        else:
            print_error(f"Repository '{repo_id}' not found.")
            print_info("Use 'list' to see available repositories.")

    def show_config(self, args: List[str] = None):
        """Show current configuration."""
        if not self.context:
            print_error("CLI context not initialized.")
            return

        config = self.context.config

        table = Table(title="⚙️ Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")

        # Show key configuration settings
        table.add_row("Neo4j URI", config.database.neo4j_uri)
        table.add_row("Neo4j Database", config.database.neo4j_database)
        table.add_row("Embedding Provider", config.embedding.embedding_provider)
        table.add_row("Embedding Model", config.embedding.model_name)
        table.add_row("Max Concurrent Files", str(config.parsing.max_concurrent_files))
        table.add_row("Batch Size", str(config.embedding.batch_size))

        self.console.print(table)

    def clear_screen(self, args: List[str] = None):
        """Clear the screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        self.show_welcome()

    def exit_shell(self, args: List[str] = None):
        """Exit the interactive shell."""
        self.running = False
        self._save_history()
        print_success("👋 Thanks for using Repository Knowledge Graph!")

    def parse_command(self, command_line: str) -> tuple[str, List[str]]:
        """Parse command line into command and arguments."""
        parts = command_line.strip().split()
        if not parts:
            return "", []
        return parts[0].lower(), parts[1:]

    async def run_command(self, command: str, args: List[str]):
        """Execute a command."""
        if command in self.commands:
            try:
                func = self.commands[command]
                if asyncio.iscoroutinefunction(func):
                    await func(args)
                else:
                    func(args)
            except KeyboardInterrupt:
                print_warning("\nCommand interrupted.")
            except Exception as e:
                print_error(f"Command failed: {e}")
        else:
            print_error(f"Unknown command: {command}")
            print_info("Type 'help' for available commands.")

    def get_prompt(self) -> str:
        """Get the current prompt string."""
        if self.current_repo_id:
            repo_short = self.current_repo_id[:8] + "..." if len(self.current_repo_id) > 11 else self.current_repo_id
            return f"[bold blue]kgraph[/bold blue]:[green]{repo_short}[/green]> "
        else:
            return "[bold blue]kgraph[/bold blue]> "

    async def run(self):
        """Main interactive loop."""
        self.show_welcome()

        while self.running:
            try:
                # Get user input
                prompt_text = self.get_prompt()
                command_line = Prompt.ask(prompt_text).strip()

                if not command_line:
                    continue

                # Parse and execute command
                command, args = self.parse_command(command_line)
                if command:
                    await self.run_command(command, args)

            except KeyboardInterrupt:
                if Confirm.ask("\n🤔 Are you sure you want to exit?", default=False):
                    self.exit_shell()
                else:
                    self.console.print()  # New line
            except EOFError:
                self.exit_shell()
            except Exception as e:
                print_error(f"Unexpected error: {e}")

    async def cleanup(self):
        """Cleanup resources."""
        if self.context:
            await self.context.cleanup()


# Main interactive CLI command
@click.command()
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file")
@click.option("--env-file", type=click.Path(exists=True), help="Path to environment file")
def interactive(config: Optional[str], env_file: Optional[str]):
    """
    Start interactive CLI shell for Repository Knowledge Graph.

    Provides a Claude Code-like interface for faster interaction.
    """
    async def run_interactive():
        cli = InteractiveCLI()
        try:
            await cli.initialize(config, env_file)
            await cli.run()
        except KeyboardInterrupt:
            print_success("\n👋 Goodbye!")
        except Exception as e:
            print_error(f"Failed to start interactive CLI: {e}")
            sys.exit(1)
        finally:
            await cli.cleanup()

    asyncio.run(run_interactive())


if __name__ == "__main__":
    interactive()