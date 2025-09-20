"""Core modules for the Repository Knowledge Graph system."""

__version__ = "1.0.0"
__author__ = "Repository Knowledge Graph Team"
__email__ = "team@repo-kgraph.com"

# Import core modules
from .parser import Parser
from .graph_builder import GraphBuilder
from .embedder import Embedder
from .retriever import Retriever
from .reducer import Reducer

__all__ = [
    "Parser",
    "GraphBuilder",
    "Embedder",
    "Retriever",
    "Reducer"
]