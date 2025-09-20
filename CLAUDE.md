# Repository Knowledge Graph System - Claude Context

**Project**: Repository to Knowledge Graph Transformation System
**Purpose**: Enable coding agents to retrieve relevant context from large codebases efficiently
**Architecture**: Python CLI + REST API with graph database and semantic search

## 🎯 Project Overview

This system transforms large code repositories into queryable knowledge graphs, solving the context limitation problem for AI coding agents (Cursor, Copilot, Gemini CLI) when working with massive codebases.

**Problem Solved**: Large repositories (10GB+, 10k+ files) exceed AI context windows, making it impossible for coding agents to understand full codebase context.

**Solution**: Build semantic knowledge graphs that enable precise context retrieval based on natural language task descriptions.

## 🏗️ System Architecture

### Core Components
- **Parser**: Tree-sitter multi-language code analysis
- **Graph Database**: Neo4j for relationship storage and traversal
- **Vector Database**: Chroma (dev) / Weaviate (prod) for semantic search
- **API Layer**: FastAPI REST endpoints for agent integration
- **CLI Interface**: Command-line tools for repository management

### Technology Stack
```yaml
Language: Python 3.10+
Parsing: Tree-sitter (primary) + Python AST (supplemental)
Graph DB: Neo4j Community → Enterprise
Vector DB: Chroma (development) → Weaviate (production)
Embeddings: Sentence Transformers → OpenAI (premium)
API: FastAPI
Testing: pytest + contract testing
Agent Framework: LangChain + selective LangGraph
```

## 📊 Data Model

### Core Entities
- **Repository**: Root container with metadata and statistics
- **CodeEntity**: Files, classes, functions, variables with semantic embeddings
- **Relationship**: Imports, calls, inheritance, dependencies with strength scores
- **KnowledgeGraph**: Complete graph structure with metrics
- **Query**: Natural language requests with processing metadata
- **ContextResult**: Ranked results with relevance scores
- **Index**: Optimized search structures for fast retrieval

### Key Relationships
```
Repository (1:N) CodeEntity
CodeEntity (N:M) Relationship
Query (1:N) ContextResult
Repository (1:1) KnowledgeGraph
```

## 🔌 API Interface

### REST Endpoints
- `POST /parse-repo`: Repository analysis and graph construction
- `POST /query`: Context retrieval for coding tasks
- `GET /graph/{entity_id}`: Relationship exploration
- `GET /repositories`: List indexed repositories
- `GET /repositories/{id}/status`: Repository status and statistics
- `GET /health`: System health monitoring

### CLI Commands
- `parse-repo PATH`: Index repository and build knowledge graph
- `query-task "DESCRIPTION"`: Retrieve relevant context
- `serve`: Start API server for agent integration
- `update PATH`: Incremental repository updates
- `status [PATH]`: Repository status and statistics
- `list`: Show all indexed repositories

## 🚀 Development Patterns

### Project Structure
```
src/
├── models/          # Data models (Repository, CodeEntity, Relationship)
├── services/        # Business logic (Parser, GraphBuilder, Retriever)
├── cli/            # Command-line interface
└── lib/            # Shared utilities

tests/
├── contract/       # API contract tests
├── integration/    # End-to-end workflow tests
└── unit/          # Component unit tests
```

### Key Development Principles
1. **Library-First**: Core components as modular libraries
2. **Test-Driven**: Write failing tests before implementation
3. **Incremental Complexity**: Start simple, add features iteratively
4. **Performance-Focused**: Optimized for large-scale processing
5. **Agent-Friendly**: MCP-compliant API design

## 🔧 Implementation Guidelines

### Code Parsing Strategy
```python
# Primary: Tree-sitter for universal language support
tree_sitter_parser = get_parser(language)
syntax_tree = tree_sitter_parser.parse(source_code)

# Supplemental: Python AST for enhanced Python analysis
if language == 'python':
    ast_tree = ast.parse(source_code)
    # Combine structural and semantic information
```

### Graph Construction Pattern
```python
# 1. Extract entities from parsed code
entities = extract_entities(syntax_tree)

# 2. Identify relationships
relationships = analyze_relationships(entities, syntax_tree)

# 3. Build graph with validation
graph = KnowledgeGraph()
graph.add_entities(entities)
graph.add_relationships(relationships)
graph.validate_constraints()
```

### Query Processing Workflow
```python
# 1. Semantic similarity search
candidates = vector_db.similarity_search(query_embedding, k=50)

# 2. Graph traversal expansion
expanded = graph_db.traverse_relationships(candidates, depth=2)

# 3. Relevance scoring and filtering
results = rank_and_filter(expanded, query, threshold=0.3)
```

## 📋 Current Implementation Status

### ✅ Completed (Phase 0-1)
- [x] Feature specification and requirements analysis
- [x] Technology research and decision making
- [x] Data model design and entity relationships
- [x] API contract specifications (OpenAPI schema)
- [x] CLI interface specification
- [x] Quickstart guide and user documentation

### 🔄 In Progress (Phase 1)
- [ ] Contract test implementations (failing tests)
- [ ] Core library structure setup
- [ ] Basic CLI command scaffolding

### ⏳ Upcoming (Phase 2-3)
- [ ] Tree-sitter integration and multi-language parsing
- [ ] Neo4j graph database integration
- [ ] Vector embedding generation and storage
- [ ] FastAPI server implementation
- [ ] End-to-end workflow testing

## 🎯 Performance Targets

### Scalability Goals
- **Repository Size**: Handle 10GB+ codebases with 10k+ files
- **Parse Time**: < 5 minutes for 1000 files
- **Query Response**: < 1 second for 90% of context queries
- **Memory Usage**: < 8GB RAM for typical repositories
- **Storage Efficiency**: < 3x original repository size

### Quality Metrics
- **Relevance Accuracy**: > 80% relevant results in top 10
- **Relationship Precision**: > 95% accurate code relationships
- **Update Performance**: < 30 seconds for incremental changes
- **System Availability**: > 99% uptime for API endpoints

## 🔍 Testing Strategy

### Test Categories
1. **Contract Tests**: API endpoint schema validation
2. **Unit Tests**: Component logic and data validation
3. **Integration Tests**: End-to-end workflow validation
4. **Performance Tests**: Scalability and response time validation

### Test-Driven Development Flow
```python
# 1. Write failing contract test
def test_parse_repository_endpoint():
    response = client.post("/parse-repo", json={"repository_path": "/test/repo"})
    assert response.status_code == 202
    assert "repository_id" in response.json()

# 2. Implement minimal functionality to pass
# 3. Refactor and optimize
# 4. Add comprehensive unit tests
```

## 🌐 Agent Integration Patterns

### MCP Compliance
- Standardized JSON response schemas
- Error handling with consistent codes
- Tool discovery through OpenAPI documentation
- Asynchronous operation support with status polling

### Example Agent Integration
```typescript
// Coding agent context retrieval
async function getCodeContext(taskDescription: string): Promise<ContextResult[]> {
  const response = await fetch('/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      repository_id: currentRepoId,
      task_description: taskDescription,
      max_results: 20,
      confidence_threshold: 0.5
    })
  });

  return (await response.json()).results;
}
```

## 🚨 Critical Implementation Notes

### Performance Considerations
- Use incremental parsing for repository updates
- Implement memory pressure monitoring and cache eviction
- Batch database operations for better performance
- Leverage parallel processing for independent file analysis

### Security & Privacy
- Validate all file paths to prevent directory traversal
- Sanitize code snippets in API responses
- Implement rate limiting for API endpoints
- Support authentication tokens for production deployments

### Error Handling
- Graceful degradation when parsing individual files fails
- Comprehensive logging for debugging complex scenarios
- Clear error messages with actionable suggestions
- Automatic recovery from transient database failures

---

**Next Phase**: Execute Phase 2 task generation using `/tasks` command to create detailed implementation tasks.