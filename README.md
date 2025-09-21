# Repository Knowledge Graph System

Transform large codebases into queryable knowledge graphs for AI-powered code understanding.

## Overview

This system enables AI coding agents to efficiently retrieve relevant context from massive codebases by transforming repositories into semantic knowledge graphs with vector embeddings. Solves the context window limitation problem for large projects.

**Key Features:**
- **Semantic Code Search**: Natural language queries to find relevant code
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, Go, Rust, and more
- **High Performance**: Optimized for 16+ core systems with concurrent processing
- **Graph Database**: Neo4j-powered relationship mapping
- **CodeBERT Embeddings**: Microsoft's code-optimized embedding model
- **REST API**: Ready for AI agent integration
- **CLI Tools**: Complete command-line interface

## Quick Start

### 1. Installation
```bash
# Clone and setup
git clone <repository-url>
cd repo-to-kgraph
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode (IMPORTANT!)
pip install -e .

# Setup databases
./kgraph setup
```

### 2. Parse a Repository
```bash
# Parse your first repository
./kgraph parse /path/to/your/project

# Expected output:
# ✓ Repository processed successfully!
# Repository ID: abc123...
# Entities: 1,247
# Relationships: 892
```

### 3. Query for Context
```bash
# Natural language queries
./kgraph query "user authentication implementation"
./kgraph query "database connection setup"
./kgraph query "API endpoints for payment processing"

# List repositories
./kgraph list

# System status
./kgraph status
```

## Configuration

The system uses a `.env` file with optimized defaults for 16-core systems:

```env
# Database Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword
CHROMA_DB_PATH=./chroma_db

# Embedding Configuration (CodeBERT)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=microsoft/codebert-base
EMBEDDING_BATCH_SIZE=128
EMBEDDING_DEVICE=cpu  # or 'cuda' for GPU

# Performance Settings (Optimized for 16-core)
MAX_CONCURRENT_FILES=32
PARSER_WORKERS=16
BATCH_SIZE=2000
MAX_FILE_SIZE_MB=25.0

# API Server
API_HOST=localhost
API_PORT=8000
```

### Performance Tuning

**High-End Systems (16+ cores):**
```env
MAX_CONCURRENT_FILES=32
EMBEDDING_BATCH_SIZE=128
PARSER_WORKERS=16
```

**Limited Resources (4-8 cores):**
```env
MAX_CONCURRENT_FILES=8
EMBEDDING_BATCH_SIZE=32
PARSER_WORKERS=4
```

**GPU Acceleration:**
```env
EMBEDDING_DEVICE=cuda
```

## CLI Commands

### Repository Management
```bash
# Parse repository
./kgraph parse /path/to/repository
./kgraph parse /path/to/repo --verbose
./kgraph parse /path/to/repo --languages python,javascript
./kgraph parse /path/to/repo --exclude "*.test.py,node_modules/*"

# Update repository (incremental)
./kgraph update /path/to/repo

# List repositories
./kgraph list

# Check status
./kgraph status [repository-id]

# Clear all repositories
./kgraph clear
```

### Querying
```bash
# Basic queries
./kgraph query "user authentication logic"
./kgraph query "database connection handling"
./kgraph query "error handling patterns"

# Advanced queries
./kgraph query "payment processing" --repository-id abc123
./kgraph query "validation logic" --max-results 20 --confidence 0.8
```

### API Server
```bash
# Start server
./kgraph serve

# Custom host/port
./kgraph serve --host 0.0.0.0 --port 8080

# Development mode
./kgraph serve --reload
```

## API Endpoints

### Parse Repository
```bash
curl -X POST http://localhost:8000/parse-repo \
  -H "Content-Type: application/json" \
  -d '{"repository_path": "/path/to/repo"}'
```

### Query for Context
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "repository_id": "abc123",
    "task_description": "implement OAuth authentication",
    "max_results": 15,
    "confidence_threshold": 0.6
  }'
```

### Other Endpoints
```bash
# List repositories
curl http://localhost:8000/repositories

# Health check
curl http://localhost:8000/health

# Repository status
curl http://localhost:8000/repositories/abc123/status
```

## Troubleshooting

### ModuleNotFoundError: No module named 'repo_kgraph'

**Error:**
```bash
./kgraph setup
# Error: ModuleNotFoundError: No module named 'repo_kgraph'
```

**Fix:**
```bash
# Install package in development mode
pip install -e .

# Verify it worked
python -c "import repo_kgraph; print('Success!')"

# Alternative if above fails
pip install --upgrade pip setuptools wheel
pip install -e .
```

**Why this happens:** The `kgraph` script runs `python -m repo_kgraph.cli` but Python needs the package to be "installed" to import it. The `-e` flag creates a link to your source code.

### Neo4j Connection Failed

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Start Neo4j if stopped
docker start neo4j-kgraph

# Check logs
docker logs neo4j-kgraph

# Verify password in .env
grep NEO4J_PASSWORD .env
```

### Performance Issues

**Slow parsing:**
```bash
# Reduce concurrent processing
echo "MAX_CONCURRENT_FILES=8" >> .env
echo "PARSER_WORKERS=4" >> .env

# Exclude large directories
./kgraph parse /path/to/repo --exclude "node_modules/*,build/*"
```

**Memory issues:**
```bash
# Reduce batch sizes
echo "EMBEDDING_BATCH_SIZE=32" >> .env
echo "BATCH_SIZE=500" >> .env
echo "MAX_FILE_SIZE_MB=10.0" >> .env
```

### Embedding Generation Fails

```bash
# Check disk space
df -h

# Clear HuggingFace cache if corrupted
rm -rf ~/.cache/huggingface/

# Fallback to CPU if GPU issues
echo "EMBEDDING_DEVICE=cpu" >> .env
```

### API Server Issues

```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
./kgraph serve --port 8080

# Reinstall package
pip install -e .
```

## Real-World Examples

### Large Project
```bash
# Configure for performance
cat >> .env << EOF
MAX_CONCURRENT_FILES=32
EMBEDDING_BATCH_SIZE=128
PARSER_WORKERS=16
EOF

# Parse with exclusions
./kgraph parse /path/to/large-project \
  --exclude "node_modules/*,build/*,dist/*,*.test.js" \
  --verbose

# Query results
./kgraph query "authentication middleware"
```

### AI Agent Integration
```bash
# Start API server
./kgraph serve --host 0.0.0.0 --port 8080

# Agent calls API
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "implement user registration with email verification",
    "max_results": 20,
    "confidence_threshold": 0.7
  }'
```

### Multiple Repositories
```bash
# Index multiple projects
./kgraph parse /path/to/frontend
./kgraph parse /path/to/backend
./kgraph parse /path/to/mobile-app

# Query across all
./kgraph query "user authentication across services"
```

## System Requirements

- **Python 3.10+**
- **Docker** (for Neo4j)
- **16GB+ RAM** recommended for large repositories
- **CUDA-compatible GPU** (optional, for faster embeddings)

## Architecture

- **Parser**: Tree-sitter + Python AST for multi-language code analysis
- **Graph Database**: Neo4j for relationship storage and traversal
- **Vector Database**: ChromaDB for semantic similarity search
- **Embeddings**: Microsoft CodeBERT for code-optimized vectors
- **API**: FastAPI REST endpoints for agent integration
- **CLI**: Click-based command-line interface