#!/usr/bin/env bash
# Development environment setup script

set -e

echo "🚀 Setting up Repository Knowledge Graph development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible (>= $required_version)"
else
    echo "❌ Python $python_version is not compatible. Please install Python >= $required_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📚 Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🎯 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p logs data/test data/cache

# Check if Neo4j is available (optional for development)
echo "🔍 Checking Neo4j availability..."
if command -v neo4j &> /dev/null; then
    echo "✅ Neo4j found in PATH"
elif docker info &> /dev/null && docker images | grep -q neo4j; then
    echo "✅ Neo4j Docker image available"
else
    echo "⚠️  Neo4j not found. You can:"
    echo "   1. Install Neo4j: https://neo4j.com/download/"
    echo "   2. Use Docker: docker run -p 7474:7474 -p 7687:7687 neo4j:latest"
    echo "   3. Use embedded/test mode for development"
fi

# Create environment configuration
if [ ! -f ".env" ]; then
    echo "⚙️  Creating development environment configuration..."
    cat > .env << EOF
# Development environment configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Database configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Vector database
CHROMA_PERSIST_DIR=./data/chroma

# API configuration
API_HOST=localhost
API_PORT=8000
API_DEBUG=true

# Embedding configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_API_KEY=your-openai-key-here

# Development settings
REPO_KGRAPH_DATA_DIR=./data
REPO_KGRAPH_LOG_FILE=./logs/app.log
EOF
    echo "✅ Environment configuration created (.env)"
    echo "   📝 Please update .env with your actual configuration values"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Update .env with your configuration"
echo "3. Start Neo4j database (if using local instance)"
echo "4. Run tests: pytest"
echo "5. Start development server: repo-kgraph serve --reload"
echo ""
echo "📚 Documentation: See docs/ directory"
echo "🔧 Run 'scripts/run-tests.sh' to execute the test suite"
echo "🚀 Run 'scripts/start-dev-server.sh' to start development server"