# Repository Knowledge Graph System

Transform large codebases into queryable knowledge graphs for AI-powered code understanding.

## Overview

This system enables AI coding agents to efficiently retrieve relevant context from massive codebases by transforming repositories into semantic knowledge graphs with vector embeddings. Solves the context window limitation problem for large projects.

**Key Features:**
- **Semantic Code Search**: Natural language queries to find relevant code
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, Go, Rust, and more
- **High Performance**: Optimized for 16+ core systems with concurrent processing
- **Unified Neo4j Storage**: Graph relationships AND vector embeddings in one database
- **Flexible Embeddings**: Ollama, Sentence Transformers, or custom providers
- **REST API**: Ready for AI agent integration
- **CLI Tools**: Complete command-line interface

## 🆕 Latest Updates (v2.0)

### Major Architecture Improvements

**🚀 Unified Neo4j Storage (Breaking Change)**
- **What Changed**: Eliminated ChromaDB dependency - now uses Neo4j for both graph relationships AND vector embeddings
- **Why**: Solves dual-database synchronization issues that caused query failures
- **Impact**: Significantly improved reliability and simplified deployment

**⚡ Enhanced Performance**
- **Atomic Operations**: Guaranteed data consistency with transaction rollback on failures
- **Faster Queries**: Direct Neo4j vector search without cross-database coordination
- **Better Scaling**: Single database connection pool for all operations

**🔧 Simplified Configuration**
- **Removed**: `CHROMA_DB_PATH` configuration (no longer needed)
- **Added**: Native Neo4j vector index support with LangChain integration
- **Default**: Ollama embeddings (`embeddinggemma:latest`) for local processing

### Migration Notes

If upgrading from v1.x:
1. **Remove ChromaDB references** from your `.env` file
2. **Reindex repositories** to use the new unified storage
3. **Update scripts** that referenced ChromaDB paths

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

### 2. Start Interactive Shell (Recommended)
```bash
# Start the interactive shell for a Claude Code-like experience
./kgraph interactive

# Inside the shell - parse your first repository
kgraph> p /path/to/your/project

# Expected output:
# ✓ Repository parsed successfully!
# Repository ID: abc123...
# Entities: 1,247
# Switched to repository: abc123
```

### 3. Query for Context
```bash
# Query for relevant code (in interactive shell)
kgraph:abc123> q user authentication
kgraph:abc123> q database connection setup
kgraph:abc123> q error handling patterns

# List repositories and switch between them
kgraph:abc123> ls
kgraph:abc123> use other-repo-id

# Check status and configuration
kgraph:abc123> status
kgraph:abc123> config

# Get help anytime
kgraph:abc123> help

# Exit when done
kgraph:abc123> exit
```

**Alternative: Traditional Commands**
```bash
# For scripts and automation, use traditional commands:
./kgraph parse /path/to/your/project
./kgraph list
./kgraph query "user authentication" --repository-id abc123
./kgraph status
```

## Configuration

The system uses a `.env` file with optimized defaults for 16-core systems:

```env
# Database Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword

# Embedding Configuration
EMBEDDING_PROVIDER=ollama  # or 'sentence_transformers'
EMBEDDING_MODEL=embeddinggemma:latest  # or 'Salesforce/SweRankEmbed-Small'
EMBEDDING_BATCH_SIZE=128
OLLAMA_CONCURRENT_REQUESTS=10
EMBEDDING_DEVICE=cpu  # or 'cuda' for GPU
```

### Performance Tuning

**High-End Systems (16+ cores):**
```env
MAX_CONCURRENT_FILES=32
EMBEDDING_BATCH_SIZE=128
OLLAMA_CONCURRENT_REQUESTS=20
PARSER_WORKERS=16
```

**Limited Resources (4-8 cores):**
```env
MAX_CONCURRENT_FILES=8
EMBEDDING_BATCH_SIZE=32
OLLAMA_CONCURRENT_REQUESTS=5
PARSER_WORKERS=4
```

**GPU Acceleration:**
```env
EMBEDDING_DEVICE=cuda
```

## 🚀 CLI Guide

### Two Ways to Use the CLI

#### 🌟 Interactive Shell (Recommended)
Start the interactive shell for a Claude Code-like experience:

```bash
# Start interactive shell
./kgraph interactive
# or use shortcuts
./kgraph i
./kgraph shell
```

**Interactive Shell Features:**
- ⚡ **Fast commands**: No need to type `./kgraph` repeatedly
- 🎯 **Context-aware**: Remembers current repository
- 📊 **Rich output**: Beautiful tables and formatted results
- 🔄 **Command history**: Persistent history with tab completion
- 💡 **Smart prompts**: Shows current repo in prompt

#### 📝 Traditional Commands
Use individual commands for scripts and automation:

```bash
./kgraph parse /path/to/repo
./kgraph query "search text"
./kgraph list
```

---

### 🎯 Interactive Shell Commands

Once in the interactive shell (`./kgraph interactive`), use these commands:

#### Repository Management
```
parse /path/to/repo    (p)     Parse and index a repository
list                   (ls)    List all indexed repositories
use <repo-id>         (switch) Switch to a different repository
status [repo-id]      (st)     Show repository status
```

#### Querying
```
query <search-text>   (q)      Query for relevant code context
```

#### Utilities
```
config                         Show current configuration
clear                 (cls)    Clear the screen
help                  (h, ?)   Show available commands
exit                  (quit)   Exit the interactive shell
```

#### Example Interactive Session
```bash
$ ./kgraph interactive

🚀 Repository Knowledge Graph - Interactive CLI
Type 'help' or '?' for available commands

kgraph> p /path/to/my-project
✓ Repository parsed successfully!
Repository ID: abc123...
Entities: 1,247
Switched to repository: abc123

kgraph:abc123> q user authentication
🔍 Results for: user authentication
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Score  ┃ Type     ┃ Name               ┃ File                         ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0.94   │ FUNCTION │ authenticate_user  │ auth/middleware.py:45        │
│ 0.89   │ CLASS    │ UserAuthenticator │ auth/authenticator.py:12     │
│ 0.84   │ FUNCTION │ verify_token      │ auth/jwt_utils.py:28         │
└────────┴──────────┴────────────────────┴──────────────────────────────┘

kgraph:abc123> q database connection
[More results...]

kgraph:abc123> ls
📚 Indexed Repositories
[Repository list...]

kgraph:abc123> help
[Command reference...]

kgraph:abc123> exit
👋 Thanks for using Repository Knowledge Graph!
```

---

### 📋 Traditional CLI Commands

For scripts, automation, or one-off commands:

#### Repository Management
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

#### Querying
```bash
# Basic queries
./kgraph query "user authentication logic"
./kgraph query "database connection handling"
./kgraph query "error handling patterns"

# Advanced queries with options
./kgraph query "payment processing" --repository-id abc123
./kgraph query "validation logic" --max-results 20 --confidence 0.8
./kgraph query "JWT authentication" --repository-id web-app --verbose
```

#### System Management
```bash
# Setup databases
./kgraph setup

# Start API server
./kgraph serve
./kgraph serve --host 0.0.0.0 --port 8080

# Run demo
./kgraph demo

# Show help
./kgraph help
```

---

### 🔧 Command Options Reference

#### Parse Command Options
| Option | Description | Example |
|--------|-------------|---------|
| `--verbose` | Enable detailed logging | `./kgraph parse /repo --verbose` |
| `--languages` | Filter by programming languages | `--languages python,javascript` |
| `--exclude` | Exclude file patterns | `--exclude "*.test.py,node_modules/*"` |
| `--repository-id` | Custom repository identifier | `--repository-id my-project` |
| `--reset-repo` | Reset and recreate repository data | `--reset-repo` |

#### Query Command Options
| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--repository-id` | Target specific repository | (all repos) | `--repository-id abc123` |
| `--max-results` | Limit number of results | 20 | `--max-results 10` |
| `--confidence` | Minimum confidence threshold | 0.3 | `--confidence 0.8` |
| `--verbose` | Show detailed processing | False | `--verbose` |

#### Server Command Options
| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--host` | Server host address | 127.0.0.1 | `--host 0.0.0.0` |
| `--port` | Server port | 8000 | `--port 8080` |
| `--reload` | Auto-reload on changes | False | `--reload` |

---

### 🚀 Getting Started Guide

#### Step 1: Setup
```bash
# Start Neo4j and setup environment
./kgraph setup
```

#### Step 2: Choose Your Interface

**For Interactive Use (Recommended):**
```bash
# Start interactive shell
./kgraph interactive

# Inside the shell:
kgraph> p /path/to/your/project
kgraph> q "what you're looking for"
```

**For Scripts and Automation:**
```bash
# Parse repository
./kgraph parse /path/to/your/project

# Query with repository ID
./kgraph query "authentication logic" --repository-id your-repo-id
```

#### Step 3: Basic Workflow

1. **Parse your repository**:
   ```bash
   ./kgraph interactive
   kgraph> p /path/to/my-project
   ```

2. **Query for relevant code**:
   ```bash
   kgraph:my-project> q user authentication
   kgraph:my-project> q database connection
   kgraph:my-project> q error handling
   ```

3. **Explore results**:
   - Click on file paths to open in your editor
   - Use results to understand code structure
   - Query for related concepts

---

### 💡 Best Practices

#### Interactive Shell Tips
- ✅ **Use shortcuts**: `p` instead of `parse`, `q` instead of `query`, `ls` instead of `list`
- ✅ **Tab completion**: Press Tab to complete commands and see options
- ✅ **Command history**: Use ↑/↓ arrows to navigate previous commands
- ✅ **Keep shell open**: Maintain session for faster repeated queries
- ✅ **Use `help`**: Type `help` anytime to see available commands

#### Query Best Practices
- 🎯 **Be specific**: "JWT authentication" vs "auth"
- 🎯 **Use natural language**: "how to connect to database" works well
- 🎯 **Adjust confidence**: Lower for more results, higher for precision
- 🎯 **Try variations**: If no results, try synonyms or broader terms

#### Repository Management
- 📁 **One-time setup**: Parse repository once, query many times
- 📁 **Update incrementally**: Use `parse` again after major code changes
- 📁 **Multiple projects**: Parse multiple repos and switch between them
- 📁 **Clean exclusions**: Exclude test files, node_modules, build directories

#### Example Workflows

**Finding Authentication Code:**
```bash
kgraph:my-app> q authentication
kgraph:my-app> q login process
kgraph:my-app> q JWT token validation
```

**Understanding Database Layer:**
```bash
kgraph:backend> q database connection
kgraph:backend> q SQL queries
kgraph:backend> q database migrations
```

**API Development:**
```bash
kgraph:api-server> q REST endpoints
kgraph:api-server> q request validation
kgraph:api-server> q error handling middleware
```

---

### 🆘 Troubleshooting

#### Interactive Shell Issues
```bash
# If shell won't start
./kgraph setup  # Ensure databases are running

# If commands don't work
help  # Check available commands
config  # Verify configuration
```

#### No Query Results
```bash
# Try lower confidence threshold
q "your search" --confidence 0.1

# Check if repository was parsed
ls  # List repositories
status  # Check current repo status

# Try broader search terms
q "auth" instead of "authentication middleware"
```

#### Repository Not Found
```bash
# List all available repositories
ls

# Parse the repository if not found
p /path/to/your/repository
```

---

## Querying Specific Repositories

### Step 1: List Available Repositories
```bash
# View all indexed repositories
./kgraph list

# Example output:
#                              Indexed Repositories
# ┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
# ┃ Repository┃ Name      ┃ Path           ┃ Entities ┃ Relations ┃ Status  ┃ Updated  ┃
# ┃ ID        ┃           ┃                ┃          ┃           ┃         ┃          ┃
# ┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
# │ my-app    │ my-app    │ /path/to/my-app│ 1,247    │ 892       │ READY   │ 2025-01- │
# │ api-server│ api-server│ /path/to/api   │ 543      │ 412       │ READY   │ 2025-01- │
# └───────────┴───────────┴────────────────┴──────────┴───────────┴─────────┴──────────┘
```

### Step 2: Query Specific Repository
```bash
# Query a specific repository by ID
./kgraph query "authentication middleware" --repository-id my-app

# Alternative: Use repository name if unique
./kgraph query "authentication middleware" --repository-id api-server

# Query with specific parameters
./kgraph query "user login flow" \
  --repository-id my-app \
  --max-results 15 \
  --confidence 0.7 \
  --verbose
```

### Step 3: Understanding Query Results
```bash
# Example query with results
./kgraph query "user authentication" --repository-id my-app

# Expected output:
# 🤔 Querying: user authentication
# ℹ Repository: my-app (/path/to/my-app)
# ℹ Found 8 relevant code contexts
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                    Query Results                                     ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ 1. authenticate_user (FUNCTION) - Score: 0.92                                       │
# │    File: auth/middleware.py:45                                                       │
# │    Description: Main user authentication function with JWT validation               │
# │                                                                                      │
# │ 2. UserAuthenticator (CLASS) - Score: 0.89                                          │
# │    File: auth/authenticator.py:12                                                    │
# │    Description: Core authentication class handling login/logout                     │
# │                                                                                      │
# │ 3. verify_token (FUNCTION) - Score: 0.84                                            │
# │    File: auth/jwt_utils.py:28                                                        │
# │    Description: JWT token verification and validation logic                         │
# └──────────────────────────────────────────────────────────────────────────────────────┘
```

### Query Options Reference

| Option | Description | Example |
|--------|-------------|---------|
| `--repository-id` | Target specific repository | `--repository-id my-app` |
| `--max-results` | Limit number of results (default: 10) | `--max-results 20` |
| `--confidence` | Minimum confidence score (0.0-1.0) | `--confidence 0.8` |
| `--no-related` | Skip related entity expansion | `--no-related` |
| `--verbose` | Show detailed processing info | `--verbose` |

### Query Strategies

#### 1. **Specific Component Queries**
```bash
# Find specific functions or classes
./kgraph query "LoginController class" --repository-id web-app
./kgraph query "password validation function" --repository-id api-server
./kgraph query "JWT token generation" --repository-id auth-service
```

#### 2. **Feature-Based Queries**
```bash
# Find code related to specific features
./kgraph query "user registration workflow" --repository-id my-app
./kgraph query "payment processing logic" --repository-id ecommerce-app
./kgraph query "file upload handling" --repository-id media-server
```

#### 3. **Architecture Queries**
```bash
# Understand system architecture
./kgraph query "database connection management" --repository-id backend
./kgraph query "API route definitions" --repository-id web-api
./kgraph query "middleware stack configuration" --repository-id express-app
```

#### 4. **Problem-Solving Queries**
```bash
# Find code for specific problems
./kgraph query "error handling patterns" --repository-id my-app
./kgraph query "input validation logic" --repository-id api-server
./kgraph query "rate limiting implementation" --repository-id web-service
```

### Advanced Query Techniques

#### Cross-Repository Analysis
```bash
# Query across multiple repositories (omit --repository-id)
./kgraph query "authentication patterns"
# This searches ALL indexed repositories for authentication-related code
```

#### Confidence Tuning
```bash
# High precision (fewer, more relevant results)
./kgraph query "OAuth implementation" --repository-id auth-app --confidence 0.9

# High recall (more results, some less relevant)
./kgraph query "OAuth implementation" --repository-id auth-app --confidence 0.3
```

#### Repository Status Check
```bash
# Check specific repository status
./kgraph status my-app

# Example output:
# ╭─────────────────────────── Repository Statistics ────────────────────────────╮
# │ Repository ID: my-app                                                         │
# │ Total Entities: 1,247                                                         │
# │ Total Relationships: 892                                                      │
# │ File Count: 156                                                               │
# │                                                                               │
# │ Language Distribution:                                                        │
# │   Python: 89 files (57.1%)                                                   │
# │   JavaScript: 45 files (28.8%)                                               │
# │   TypeScript: 22 files (14.1%)                                               │
# │                                                                               │
# │ Embedding Info:                                                               │
# │   Provider: ollama                                                            │
# │   Model: embeddinggemma:latest                                                │
# │   Dimension: 768                                                              │
# │   Storage: Neo4j Vector Index                                                 │
# ╰───────────────────────────────────────────────────────────────────────────────╯
```

### Troubleshooting Queries

#### No Results Found
```bash
# If you get no results, try:
1. Lower confidence threshold:
   ./kgraph query "your search" --repository-id repo --confidence 0.1

2. Check repository status:
   ./kgraph status repo

3. Verify repository is indexed:
   ./kgraph list

4. Try broader search terms:
   ./kgraph query "auth" instead of "authentication middleware"
```

#### Repository Not Found
```bash
# Error: Repository 'my-repo' not found
# Solution: Check exact repository ID
./kgraph list
./kgraph query "search term" --repository-id exact-repo-id-from-list
```

### Best Practices

1. **Always start with `./kgraph list`** to see available repositories
2. **Use specific repository IDs** for focused searches
3. **Start with broader terms** then narrow down
4. **Adjust confidence** based on results quality
5. **Check repository status** if queries seem incomplete

### Quick Reference: Common Query Examples

```bash
# Repository Management
./kgraph list                                    # List all repositories
./kgraph status my-app                          # Check repository status
./kgraph parse /path/to/repo --repository-id my-app  # Index repository

# Basic Queries
./kgraph query "authentication" --repository-id my-app
./kgraph query "database connection" --repository-id backend
./kgraph query "error handling" --repository-id api-server

# Advanced Queries
./kgraph query "JWT validation" --repository-id auth --confidence 0.8
./kgraph query "user registration" --repository-id web-app --max-results 20
./kgraph query "payment processing" --repository-id ecommerce --verbose

# Cross-Repository Search
./kgraph query "authentication patterns"        # Search all repositories
./kgraph query "database connection" --confidence 0.5  # Lower threshold for more results
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
- **Unified Storage**: Neo4j for both graph relationships and vector embeddings
- **Vector Search**: Neo4j Vector Index with LangChain integration
- **Embeddings**: Ollama (default) or Sentence Transformers for code-optimized vectors
- **API**: FastAPI REST endpoints for agent integration
- **CLI**: Click-based command-line interface

### Key Improvements in Latest Version

- **🚀 Unified Storage**: Eliminated dual-database complexity by storing both graph data and vector embeddings in Neo4j
- **⚡ Better Performance**: Reduced synchronization issues and improved query reliability
- **🔧 Simplified Configuration**: No more ChromaDB setup required - just Neo4j
- **📈 Enhanced Reliability**: Atomic operations ensure data consistency
- **🎯 Ollama Integration**: Local embedding generation with `embeddinggemma:latest`