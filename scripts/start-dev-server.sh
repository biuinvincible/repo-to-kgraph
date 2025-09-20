#!/usr/bin/env bash
# Development server startup script

set -e

echo "🚀 Starting Repository Knowledge Graph Development Server"
echo "======================================================="

# Activate virtual environment if available
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "⚙️  Loading environment variables from .env"
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Default configuration
HOST=${API_HOST:-localhost}
PORT=${API_PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-info}
RELOAD=${RELOAD:-true}

echo "🌐 Server configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log Level: $LOG_LEVEL"
echo "   Auto-reload: $RELOAD"
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use!"
    echo "   Kill the process or use a different port"
    echo "   To kill: lsof -ti :$PORT | xargs kill -9"
    exit 1
fi

# Start the development server
echo "🔥 Starting FastAPI development server..."
echo "   API docs: http://$HOST:$PORT/docs"
echo "   Health check: http://$HOST:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

if [ "$RELOAD" = "true" ]; then
    uvicorn repo_kgraph.api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --reload \
        --reload-dir src
else
    uvicorn repo_kgraph.api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL"
fi