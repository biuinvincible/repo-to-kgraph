"""
Code parser service using Tree-sitter for multi-language parsing.

Handles extraction of code entities and relationships from source files
using Tree-sitter for universal language support with Python AST fallback.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass

try:
    import tree_sitter_languages
    from tree_sitter import Language, Parser, Node, Tree
except ImportError:
    tree_sitter_languages = None
    Language = None
    Parser = None
    Node = None
    Tree = None

from repo_kgraph.models.code_entity import CodeEntity, EntityType
from repo_kgraph.models.relationship import Relationship, RelationshipType


logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a single file."""
    file_path: str
    entities: List[CodeEntity]
    relationships: List[Relationship]
    parse_time_ms: float
    success: bool
    error_message: Optional[str] = None
    language: Optional[str] = None


@dataclass
class ParsingStats:
    """Statistics from parsing operation."""
    total_files: int
    parsed_files: int
    failed_files: int
    total_entities: int
    total_relationships: int
    parse_time_seconds: float
    errors: List[str]


class LanguageNotSupportedError(Exception):
    """Raised when a language is not supported by the parser."""
    pass


class CodeParser:
    """
    Multi-language code parser using Tree-sitter.

    Provides unified interface for parsing source code across different
    programming languages with entity and relationship extraction.
    """

    def __init__(self, max_file_size_mb: float = 10.0):
        """
        Initialize the code parser.

        Args:
            max_file_size_mb: Maximum file size to parse in megabytes
        """
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self._parsers: Dict[str, Parser] = {}
        self._languages: Dict[str, Language] = {}
        self._initialize_parsers()

    def _initialize_parsers(self) -> None:
        """Initialize Tree-sitter parsers for supported languages."""
        if not tree_sitter_languages:
            logger.warning("Tree-sitter not available, falling back to basic parsing")
            return

        # Language mapping from file extensions to Tree-sitter language names
        self.language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".cc": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "c_sharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".clj": "clojure",
            ".hs": "haskell",
            ".ml": "ocaml",
            ".fs": "fsharp",
            ".lua": "lua",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",

            # Config and Infrastructure Files
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".ini": "ini",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",

            # Infrastructure as Code
            ".tf": "hcl",           # Terraform
            ".hcl": "hcl",          # HashiCorp Configuration Language
            ".dockerfile": "dockerfile",

            # Other markup and data
            ".md": "markdown",
            ".rst": "rst",
            ".tex": "latex",
        }

        # Initialize available parsers
        for ext, lang_name in self.language_map.items():
            try:
                language = tree_sitter_languages.get_language(lang_name)
                parser = Parser()
                parser.set_language(language)
                self._languages[lang_name] = language
                self._parsers[lang_name] = parser
                logger.debug(f"Initialized parser for {lang_name}")
            except Exception as e:
                logger.debug(f"Could not initialize parser for {lang_name}: {e}")

    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return list(self._languages.keys())

    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file extension and special file names.

        Args:
            file_path: Path to the source file

        Returns:
            Language name if supported, None otherwise
        """
        path = Path(file_path)
        file_name = path.name.lower()
        extension = path.suffix.lower()

        # Special file name mappings (files without extensions)
        special_files = {
            'dockerfile': 'dockerfile',
            'docker-compose.yml': 'yaml',
            'docker-compose.yaml': 'yaml',
            'makefile': 'makefile',
            'rakefile': 'ruby',
            'gemfile': 'ruby',
            'podfile': 'ruby',
            'vagrantfile': 'ruby',
            'cmakelists.txt': 'cmake',
            '.gitignore': 'gitignore',
            '.dockerignore': 'gitignore',
            
            # Kubernetes files (common patterns)
            'deployment.yaml': 'yaml',
            'service.yaml': 'yaml',
            'configmap.yaml': 'yaml',
            'secret.yaml': 'yaml',
            'ingress.yaml': 'yaml',
            'deployment.yml': 'yaml',
            'service.yml': 'yaml',
            'configmap.yml': 'yaml',
            'secret.yml': 'yaml',
            'ingress.yml': 'yaml',
        }

        # Check special file names first
        if file_name in special_files:
            lang = special_files[file_name]
            # Only return if we have a parser for this language
            if lang in self._parsers:
                return lang

        # Check for Kubernetes files (common patterns)
        if file_name.endswith(('-deployment.yaml', '-service.yaml', '-configmap.yaml', '-secret.yaml', '-ingress.yaml',
                              '-deployment.yml', '-service.yml', '-configmap.yml', '-secret.yml', '-ingress.yml')):
            if 'yaml' in self._parsers:
                return 'yaml'

        # Then check extensions, only return if we have a parser
        lang = self.language_map.get(extension)
        if lang and lang in self._parsers:
            return lang

        return None

    async def parse_file(self, file_path: str, repository_id: str, repository_path: Optional[str] = None) -> ParseResult:
        """
        Parse a single source file.

        Args:
            file_path: Path to the source file
            repository_id: Repository identifier for entities

        Returns:
            ParseResult with entities and relationships
        """
        start_time = asyncio.get_event_loop().time()

        # Calculate relative path for CodeEntity objects
        path = Path(file_path).resolve()
        if repository_path:
            try:
                repo_path = Path(repository_path).resolve()
                relative_file_path = str(path.relative_to(repo_path))
                logger.debug(f"Relative path calculated: {relative_file_path} from {file_path} relative to {repository_path}")
            except ValueError as e:
                # If path is not relative to repository_path, use just the filename
                logger.warning(f"Failed to calculate relative path for {file_path} relative to {repository_path}: {e}")
                relative_file_path = path.name
        else:
            relative_file_path = path.name

        try:
            # Validate file
            if not path.exists():
                return ParseResult(
                    file_path=relative_file_path,
                    entities=[],
                    relationships=[],
                    parse_time_ms=0.0,
                    success=False,
                    error_message=f"File not found: {file_path}"
                )

            # Check file size
            if path.stat().st_size > self.max_file_size_bytes:
                return ParseResult(
                    file_path=relative_file_path,
                    entities=[],
                    relationships=[],
                    parse_time_ms=0.0,
                    success=False,
                    error_message=f"File too large: {path.stat().st_size} bytes"
                )

            # Detect language
            language = self.detect_language(file_path)
            if not language:
                return ParseResult(
                    file_path=relative_file_path,
                    entities=[],
                    relationships=[],
                    parse_time_ms=0.0,
                    success=False,
                    error_message=f"Unsupported file type: {path.suffix}"
                )

            # Read file content
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = path.read_text(encoding='latin-1')
                except Exception as e:
                    return ParseResult(
                        file_path=relative_file_path,
                        entities=[],
                        relationships=[],
                        parse_time_ms=0.0,
                        success=False,
                        error_message=f"Could not read file: {e}"
                    )

            # Parse with Tree-sitter
            entities, relationships = await self._parse_with_tree_sitter(
                content, file_path, relative_file_path, language, repository_id
            )

            end_time = asyncio.get_event_loop().time()
            parse_time_ms = (end_time - start_time) * 1000

            return ParseResult(
                file_path=relative_file_path,
                entities=entities,
                relationships=relationships,
                parse_time_ms=parse_time_ms,
                success=True,
                language=language
            )

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            parse_time_ms = (end_time - start_time) * 1000

            logger.error(f"Error parsing {file_path}: {e}")
            return ParseResult(
                file_path=relative_file_path,
                entities=[],
                relationships=[],
                parse_time_ms=parse_time_ms,
                success=False,
                error_message=str(e)
            )

    async def _parse_with_tree_sitter(
        self,
        content: str,
        file_path: str,
        relative_file_path: str,
        language: str,
        repository_id: str
    ) -> Tuple[List[CodeEntity], List[Relationship]]:
        """Parse content using Tree-sitter."""
        if language not in self._parsers:
            raise LanguageNotSupportedError(f"No parser available for {language}")

        parser = self._parsers[language]
        tree = parser.parse(bytes(content, "utf-8"))

        entities = []
        relationships = []

        # Extract entities based on language
        if language == "python":
            entities, relationships = self._extract_python_entities(
                tree.root_node, content, file_path, relative_file_path, repository_id
            )
        elif language in ["javascript", "typescript"]:
            entities, relationships = self._extract_js_entities(
                tree.root_node, content, file_path, relative_file_path, repository_id
            )
        else:
            # Generic extraction for other languages
            entities, relationships = self._extract_generic_entities(
                tree.root_node, content, file_path, relative_file_path, repository_id, language
            )

        return entities, relationships

    def _extract_python_entities(
        self,
        root: Any,
        content: str,
        file_path: str,
        relative_file_path: str,
        repository_id: str
    ) -> Tuple[List[CodeEntity], List[Relationship]]:
        """Extract entities from Python code."""
        entities = []
        relationships = []
        lines = content.split('\n')
        
        def extract_node(node: Any, parent_id: Optional[str] = None) -> None:
            node_type = getattr(node, 'type', None)
            logger.debug(f"Processing node of type: {node_type}")
            
            if node_type == 'function_definition':
                logger.debug("Found function definition node")
                func_entity = self._create_function_entity(
                    node, content, file_path, relative_file_path, repository_id, "python"
                )
                # Debug: Check if func_entity is None
                if func_entity is None:
                    logger.warning("Failed to create function entity")
                    return

                entities.append(func_entity)

                # Add containment relationship if within a class
                if parent_id:
                    rel = Relationship(
                        source_entity_id=parent_id,
                        target_entity_id=func_entity.id,
                        relationship_type=RelationshipType.CONTAINS,
                        line_number=func_entity.start_line
                    )
                    relationships.append(rel)

                # Process function body for calls
                # Recurse into children of the function body using the function as parent
                for child in getattr(node, 'children', []):
                    try:
                        extract_node(child, func_entity.id)
                    except Exception:
                        logger.exception("Error extracting child node inside function %s", getattr(func_entity, 'name', '<unknown>'))

            elif node_type == 'class_definition':
                logger.debug("Found class definition node")
                class_entity = self._create_class_entity(
                    node, content, file_path, relative_file_path, repository_id, "python"
                )
                # Debug: Check if class_entity is None
                if class_entity is None:
                    logger.warning("Failed to create class entity")
                    return

                entities.append(class_entity)

                # Process class body
                # Recurse into class children
                for child in getattr(node, 'children', []):
                    try:
                        extract_node(child, class_entity.id)
                    except Exception:
                        logger.exception("Error extracting child node inside class %s", getattr(class_entity, 'name', '<unknown>'))

            elif node_type == 'call':
                # Extract function calls for relationships
                if parent_id:
                    # This would need more sophisticated analysis
                    # to resolve function names to entity IDs
                    pass

            # Recursively process ALL children to ensure we find nested entities
            # We process specific node types above, then recurse into all children
            for child in getattr(node, 'children', []):
                try:
                    extract_node(child, parent_id)
                except Exception:
                    child_type = getattr(child, 'type', '<unknown>')
                    logger.exception("Error extracting child node %s", child_type)

        # Start extraction
        logger.debug("Starting node extraction")
        extract_node(root)
        logger.debug(f"Extraction complete. Found {len(entities)} entities and {len(relationships)} relationships")

        # Add file entity
        file_entity = CodeEntity(
            repository_id=repository_id,
            entity_type=EntityType.FILE,
            name=Path(file_path).name,
            qualified_name=file_path,
            file_path=relative_file_path,
            start_line=1,
            end_line=len(lines),
            start_column=0,
            end_column=0,
            language="python",
            line_count=len(lines)
        )
        entities.insert(0, file_entity)

        return entities, relationships

    def _extract_js_entities(
        self,
        root: Any,
        content: str,
        file_path: str,
        relative_file_path: str,
        repository_id: str
    ) -> Tuple[List[CodeEntity], List[Relationship]]:
        """Extract entities from JavaScript/TypeScript code."""
        entities = []
        relationships = []
        lines = content.split('\n')

        # Similar to Python extraction but for JS/TS syntax
        # This would be implemented with JS-specific node types

        # Add file entity
        file_entity = CodeEntity(
            repository_id=repository_id,
            entity_type=EntityType.FILE,
            name=Path(file_path).name,
            qualified_name=file_path,
            file_path=relative_file_path,
            start_line=1,
            end_line=len(lines),
            start_column=0,
            end_column=0,
            language="javascript",
            line_count=len(lines)
        )
        entities.append(file_entity)

        return entities, relationships

    def _extract_generic_entities(
        self,
        root: Any,
        content: str,
        file_path: str,
        relative_file_path: str,
        repository_id: str,
        language: str
    ) -> Tuple[List[CodeEntity], List[Relationship]]:
        """Generic entity extraction for other languages."""
        entities = []
        relationships = []
        lines = content.split('\n')

        # Basic file entity for unsupported languages
        file_entity = CodeEntity(
            repository_id=repository_id,
            entity_type=EntityType.FILE,
            name=Path(file_path).name,
            qualified_name=file_path,
            file_path=relative_file_path,
            start_line=1,
            end_line=len(lines),
            start_column=0,
            end_column=0,
            language=language,
            line_count=len(lines)
        )
        entities.append(file_entity)

        return entities, relationships

    def _create_function_entity(
        self,
        node: Any,
        content: str,
        file_path: str,
        relative_file_path: str,
        repository_id: str,
        language: str
    ) -> CodeEntity:
        """Create a function entity from Tree-sitter node."""
        try:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            start_column = node.start_point[1]
            end_column = node.end_point[1]

            # Extract function name (simplified)
            func_name = "unknown_function"
            for child in getattr(node, 'children', []):
                if getattr(child, 'type', None) == 'identifier':
                    # Use node.text in new API
                    func_name = child.text.decode('utf-8') if getattr(child, 'text', None) else 'unknown_function'
                    break

            # Extract signature (simplified)
            signature = node.text.decode('utf-8').split('\n')[0] if getattr(node, 'text', None) else ""

            entity = CodeEntity(
                repository_id=repository_id,
                entity_type=EntityType.FUNCTION,
                name=func_name,
                qualified_name=f"{Path(relative_file_path).stem}.{func_name}",
                file_path=relative_file_path,
                start_line=start_line,
                end_line=end_line,
                start_column=start_column,
                end_column=end_column,
                language=language,
                signature=signature,
                line_count=end_line - start_line + 1
            )
            
            logger.debug(f"Created function entity: {func_name}")
            return entity
            
        except Exception as e:
            logger.error(f"Error creating function entity: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _create_class_entity(
        self,
        node: Node,
        content: str,
        file_path: str,
        relative_file_path: str,
        repository_id: str,
        language: str
    ) -> CodeEntity:
        """Create a class entity from Tree-sitter node."""
        try:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            start_column = node.start_point[1]
            end_column = node.end_point[1]

            # Extract class name (simplified)
            class_name = "unknown_class"
            for child in node.children:
                if child.type == 'identifier':
                    class_name = child.text.decode('utf-8')
                    break

            entity = CodeEntity(
                repository_id=repository_id,
                entity_type=EntityType.CLASS,
                name=class_name,
                qualified_name=f"{Path(file_path).stem}.{class_name}",
                file_path=relative_file_path,
                start_line=start_line,
                end_line=end_line,
                start_column=start_column,
                end_column=end_column,
                language=language,
                line_count=end_line - start_line + 1
            )
            
            logger.debug(f"Created class entity: {class_name}")
            return entity
            
        except Exception as e:
            logger.error(f"Error creating class entity: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


    async def parse_directory(
        self,
        directory_path: str,
        repository_id: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_workers: int = 4
    ) -> ParsingStats:
        """
        Parse all supported files in a directory.

        Args:
            directory_path: Path to directory to parse
            repository_id: Repository identifier
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            max_workers: Maximum concurrent parsing tasks

        Returns:
            ParsingStats with parsing results
        """
        start_time = asyncio.get_event_loop().time()

        # Find files to parse
        files_to_parse = self._find_files_to_parse(
            directory_path, include_patterns, exclude_patterns
        )

        # Parse files concurrently
        semaphore = asyncio.Semaphore(max_workers)
        tasks = []

        async def parse_with_semaphore(file_path: str) -> ParseResult:
            async with semaphore:
                # Convert relative path to absolute path for file reading
                absolute_path = str(Path(directory_path) / file_path)
                return await self.parse_file(absolute_path, repository_id, directory_path)

        for file_path in files_to_parse:
            task = asyncio.create_task(parse_with_semaphore(file_path))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Compile statistics and collect entities
        parsed_files = 0
        failed_files = 0
        total_entities = 0
        total_relationships = 0
        errors = []
        all_entities = []
        all_relationships = []

        for result in results:
            if isinstance(result, Exception):
                failed_files += 1
                errors.append(str(result))
            elif isinstance(result, ParseResult):
                if result.success:
                    parsed_files += 1
                    total_entities += len(result.entities)
                    total_relationships += len(result.relationships)
                    # Collect entities and relationships
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)
                else:
                    failed_files += 1
                    if result.error_message:
                        errors.append(f"{result.file_path}: {result.error_message}")

        end_time = asyncio.get_event_loop().time()
        parse_time_seconds = end_time - start_time

        return ParsingStats(
            total_files=len(files_to_parse),
            parsed_files=parsed_files,
            failed_files=failed_files,
            total_entities=total_entities,
            total_relationships=total_relationships,
            parse_time_seconds=parse_time_seconds,
            errors=errors
        )

    async def stream_directory_entities(
        self,
        directory_path: str,
        repository_id: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        entity_callback: Optional[callable] = None,
        relationship_callback: Optional[callable] = None
    ) -> ParsingStats:
        """
        Stream entities and relationships as they are parsed, processing them immediately.

        Args:
            directory_path: Path to directory to parse
            repository_id: Repository identifier
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            entity_callback: Optional callback for each entity (async function)
            relationship_callback: Optional callback for each relationship (async function)

        Returns:
            ParsingStats with counts only (no stored entities)
        """
        start_time = asyncio.get_event_loop().time()

        # Find files to parse
        files_to_parse = self._find_files_to_parse(directory_path, include_patterns, exclude_patterns)

        # Initialize counters
        parsed_files = 0
        failed_files = 0
        total_entities = 0
        total_relationships = 0
        errors = []

        # Process files one by one, streaming results
        for file_path in files_to_parse:
            try:
                result = await self.parse_file(file_path, repository_id, directory_path)

                if result.success:
                    parsed_files += 1
                    total_entities += len(result.entities)
                    total_relationships += len(result.relationships)

                    # Stream entities immediately to callback
                    if entity_callback:
                        for entity in result.entities:
                            await entity_callback(entity)

                    # Stream relationships immediately to callback
                    if relationship_callback:
                        for relationship in result.relationships:
                            await relationship_callback(relationship)

                else:
                    failed_files += 1
                    if result.error_message:
                        errors.append(f"{result.file_path}: {result.error_message}")

            except Exception as e:
                failed_files += 1
                errors.append(f"{file_path}: {str(e)}")
                logger.error(f"Error parsing file {file_path}: {e}")

        end_time = asyncio.get_event_loop().time()
        parse_time_seconds = end_time - start_time

        return ParsingStats(
            total_files=len(files_to_parse),
            parsed_files=parsed_files,
            failed_files=failed_files,
            total_entities=total_entities,
            total_relationships=total_relationships,
            parse_time_seconds=parse_time_seconds,
            errors=errors
        )

    def _find_files_to_parse(
        self,
        directory_path: str,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[str]:
        """Find files to parse in directory."""
        import fnmatch

        directory = Path(directory_path)
        files_to_parse = []

        exclude_patterns = exclude_patterns or []
        default_excludes = [
            "*.pyc",
            "__pycache__/*",
            "node_modules/*",
            "*/node_modules/*",
            ".git/*",
            "*.min.js",
            "*.min.css",
            "build/*",
            "dist/*",
            ".venv/*",
            "venv/*",
            "env/*",
            ".env/*",
            "*.egg-info/*",
            ".pytest_cache/*",
            ".idea/*",
            ".vscode/*",
            "*.db",
            "checkpoints.db",
            "*.log",
            "logs/*",
        ]
        exclude_patterns.extend(default_excludes)

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            # Check if file type is supported
            if not self.detect_language(str(file_path)):
                continue

            relative_path = str(file_path.relative_to(directory))

            # Check exclude patterns
            excluded = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(relative_path, pattern):
                    excluded = True
                    break

            if excluded:
                continue

            # Check include patterns if specified
            if include_patterns:
                included = False
                for pattern in include_patterns:
                    if fnmatch.fnmatch(relative_path, pattern):
                        included = True
                        break
                if not included:
                    continue

            files_to_parse.append(str(file_path))

        return files_to_parse

    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""