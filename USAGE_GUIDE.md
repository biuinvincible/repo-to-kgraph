# Repository Knowledge Graph - Complete Usage Guide

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
nano .env  # Edit with your settings

# Start Neo4j database
./kgraph setup
```

### 2. Parse Your First Repository
```bash
# Parse a repository
./kgraph parse /mnt/d/agentic-chatbot

# Expected output:
# 🔧 Loading configuration from .env file...
# 🔍 Parsing repository: /mnt/d/agentic-chatbot
# ✓ Repository processed successfully!
# Repository ID: abc123...
# Entities: 150
# Relationships: 45
```

### 3. Query for Context
```bash
# Get repository list
./kgraph list

# Query for specific functionality
./kgraph query "user authentication" --repository-id abc123
./kgraph query "database connection setup"
./kgraph query "API endpoints for user management"
```

---

## ⚙️ Configuration

### Environment Variables (.env file)

The system uses a `.env` file for all configuration. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

#### Key Settings:

**Database Configuration:**
```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword  # CHANGE THIS!
NEO4J_DATABASE=neo4j
CHROMA_DB_PATH=./chroma_db
```

**Embedding Configuration:**
```env
EMBEDDING_PROVIDER=ollama  # or: sentence_transformers, openai
EMBEDDING_MODEL=embeddinggemma
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_TEXT_LENGTH=8192

# If using OpenAI
# OPENAI_API_KEY=your_api_key_here
```

**Performance Settings:**
```env
MAX_CONCURRENT_FILES=10
BATCH_SIZE=1000
MAX_FILE_SIZE_MB=10.0
```

**API Server:**
```env
API_HOST=localhost
API_PORT=8000
API_RELOAD=false
```

**Parsing Configuration:**
```env
# Languages to include (empty = all)
# INCLUDE_LANGUAGES=python,javascript,typescript,java

# File patterns to exclude
EXCLUDE_PATTERNS=*.pyc,__pycache__/*,node_modules/*,*.min.js,build/*,dist/*,.git/*

# Processing options
INCREMENTAL_PARSING=false
GENERATE_EMBEDDINGS=true
```

---

## 📋 CLI Commands

### Repository Management

#### Parse Repository
```bash
# Basic parsing
./kgraph parse /path/to/repository

# With options
./kgraph parse /path/to/repo --verbose
./kgraph parse /path/to/repo --languages python,javascript
./kgraph parse /path/to/repo --exclude "*.test.py,node_modules/*"

# Incremental update
./kgraph update /path/to/repo
```

#### List & Status
```bash
# List all repositories
./kgraph list

# Check repository status
./kgraph status REPO_ID

# Check system status
./kgraph status
```

### Query & Search

#### Basic Queries
```bash
# Natural language queries
./kgraph query "implement user authentication"
./kgraph query "database connection logic"
./kgraph query "API endpoints for payments"
```

#### Advanced Queries
```bash
# With specific repository
./kgraph query "user login" --repository-id abc123

# With filters
./kgraph query "error handling" --max-results 20 --confidence 0.7
./kgraph query "data validation" --language python --entity-type FUNCTION
```

### API Server

#### Start Server
```bash
# Basic server
./kgraph serve

# Custom host/port
./kgraph serve --host 0.0.0.0 --port 8080

# Development mode with auto-reload
./kgraph serve --reload
```

#### API Endpoints
Once running, the API provides:

```bash
# Parse repository
curl -X POST http://localhost:8000/parse-repo \
  -H "Content-Type: application/json" \
  -d '{"repository_path": "/path/to/repo"}'

# Query for context
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "repository_id": "abc123",
    "task_description": "implement user authentication",
    "max_results": 10
  }'

# Health check
curl http://localhost:8000/health
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.10+
- Docker (for Neo4j)
- Git

### Full Installation
```bash
# 1. Clone repository
git clone <repository-url>
cd repo-to-kgraph

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
nano .env  # Edit NEO4J_PASSWORD and other settings

# 5. Setup databases
./kgraph setup

# 6. Test installation
./kgraph --help
```

### Docker Alternative
```bash
# Start Neo4j manually
docker run --name neo4j-kgraph \
  -p7474:7474 -p7687:7687 \
  --env NEO4J_AUTH=neo4j/testpassword \
  -d neo4j:latest

# Verify Neo4j is running
curl http://localhost:7474
```

---

## 🎯 Real-World Examples

### Example 1: Index a Large Project
```bash
# Configure for large repository
cat >> .env << EOF
MAX_CONCURRENT_FILES=20
BATCH_SIZE=2000
MAX_FILE_SIZE_MB=20.0
EOF

# Parse with exclusions
./kgraph parse /path/to/large-project \
  --exclude "node_modules/*,build/*,dist/*,*.test.js" \
  --verbose

# Query the results
./kgraph query "authentication middleware"
```

### Example 2: AI Agent Integration
```bash
# Start API server
./kgraph serve --host 0.0.0.0 --port 8080

# Your AI agent calls:
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "repository_id": "abc123",
    "task_description": "I need to add a new user registration endpoint",
    "max_results": 15,
    "confidence_threshold": 0.6
  }'
```

### Example 3: Multiple Repositories
```bash
# Parse multiple projects
./kgraph parse /path/to/frontend
./kgraph parse /path/to/backend
./kgraph parse /path/to/mobile-app

# List all
./kgraph list

# Query across all repositories
./kgraph query "user authentication implementation"
```

---

## 🛠️ Troubleshooting

### Common Issues

#### Neo4j Connection Failed
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Start if stopped
docker start neo4j-kgraph

# Check logs
docker logs neo4j-kgraph

# Verify password in .env
grep NEO4J_PASSWORD .env
```

#### Parsing Too Slow
```bash
# Reduce concurrent files
echo "MAX_CONCURRENT_FILES=4" >> .env

# Exclude large directories
./kgraph parse /path/to/repo --exclude "node_modules/*,build/*"

# Check disk space
df -h
```

#### Embedding Generation Fails
```bash
# Check Ollama is running (if using ollama provider)
curl http://localhost:11434/api/tags

# Or switch to sentence transformers
echo "EMBEDDING_PROVIDER=sentence_transformers" >> .env
echo "EMBEDDING_MODEL=all-MiniLM-L6-v2" >> .env
```

#### API Server Won't Start
```bash
# Check port availability
netstat -tulpn | grep 8000

# Use different port
./kgraph serve --port 8080

# Check logs
./kgraph serve --verbose
```

### Performance Tuning

#### For Large Repositories (10k+ files)
```env
# In .env file
MAX_CONCURRENT_FILES=20
BATCH_SIZE=2000
MAX_FILE_SIZE_MB=20.0
EMBEDDING_BATCH_SIZE=64
```

#### For Limited Resources
```env
# In .env file
MAX_CONCURRENT_FILES=2
BATCH_SIZE=500
MAX_FILE_SIZE_MB=5.0
EMBEDDING_BATCH_SIZE=16
```

---

## 📈 Advanced Usage

### Custom Language Support
Add new file extensions by modifying the parser configuration or contributing to the tree-sitter language support.

### Custom Embedding Models
```env
# Sentence Transformers
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-mpnet-base-v2

# OpenAI
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_API_KEY=your_api_key
```

### Production Deployment
```env
# Production settings
API_HOST=0.0.0.0
API_PORT=80
LOG_LEVEL=WARNING
NEO4J_URI=neo4j://neo4j-server:7687
CHROMA_DB_PATH=/data/chroma_db
```

---

## 🎉 That's It!

The Repository Knowledge Graph system is now ready to help your AI coding agents efficiently retrieve relevant context from large codebases. Happy coding! 🚀