#!/usr/bin/env bash
# Test execution script with different test categories

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Repository Knowledge Graph Test Suite${NC}"
echo "========================================"

# Default to all tests if no argument provided
TEST_TYPE=${1:-all}

# Activate virtual environment if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

case $TEST_TYPE in
    "unit")
        echo -e "${BLUE}Running unit tests...${NC}"
        pytest tests/unit/ -v --tb=short
        ;;

    "integration")
        echo -e "${BLUE}Running integration tests...${NC}"
        pytest tests/integration/ -v --tb=short
        ;;

    "contract")
        echo -e "${BLUE}Running contract tests...${NC}"
        pytest tests/contract/ -v --tb=short
        ;;

    "performance")
        echo -e "${BLUE}Running performance tests...${NC}"
        pytest tests/performance/ -v --tb=short -m "not slow"
        ;;

    "slow")
        echo -e "${YELLOW}Running slow tests (this may take a while)...${NC}"
        pytest tests/performance/ -v --tb=short -m slow
        ;;

    "fast")
        echo -e "${BLUE}Running fast tests (excluding slow and integration)...${NC}"
        pytest tests/ -v --tb=short -m "not slow and not integration"
        ;;

    "coverage")
        echo -e "${BLUE}Running tests with coverage report...${NC}"
        pytest --cov=src --cov-report=html --cov-report=term-missing
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;

    "ci")
        echo -e "${BLUE}Running CI test suite...${NC}"
        pytest --cov=src --cov-report=xml --cov-fail-under=80 -m "not slow"
        ;;

    "all")
        echo -e "${BLUE}Running complete test suite...${NC}"
        pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
        ;;

    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo ""
        echo "Usage: $0 [test_type]"
        echo ""
        echo "Available test types:"
        echo "  unit        - Unit tests only"
        echo "  integration - Integration tests only"
        echo "  contract    - Contract tests only"
        echo "  performance - Performance tests (excluding slow)"
        echo "  slow        - Slow performance tests"
        echo "  fast        - Fast tests (excluding slow and integration)"
        echo "  coverage    - All tests with HTML coverage report"
        echo "  ci          - CI test suite (no slow tests, XML coverage)"
        echo "  all         - All tests (default)"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Test execution completed!${NC}"