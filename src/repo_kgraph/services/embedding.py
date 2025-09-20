"""
Embedding service for generating semantic vectors.

Handles text embedding generation using Ollama with embeddinggemma model,
sentence transformers, and OpenAI with vector storage in Chroma database 
for similarity search.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    SentenceTransformer = None
    cos_sim = None

try:
    import openai
except ImportError:
    openai = None

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

from repo_kgraph.models.code_entity import CodeEntity
from repo_kgraph.models.query import Query


logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingService:
    """
    Service for generating and managing semantic embeddings.

    Supports multiple embedding models and vector storage backends
    with optimized batch processing and similarity search.
    """

    def __init__(
        self,
        model_name: str = "embeddinggemma",
        embedding_provider: str = "ollama",  # or "sentence_transformers" or "openai"
        chroma_db_path: str = "./chroma_db",
        openai_api_key: Optional[str] = None,
        batch_size: int = 32,
        max_text_length: int = 8192
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model to use
            embedding_provider: Provider for embeddings ("ollama", "sentence_transformers" or "openai")
            chroma_db_path: Path to Chroma database storage
            openai_api_key: OpenAI API key if using OpenAI embeddings
            batch_size: Batch size for processing
            max_text_length: Maximum text length for embeddings
        """
        self.model_name = model_name
        self.embedding_provider = embedding_provider
        self.chroma_db_path = chroma_db_path
        self.batch_size = batch_size
        self.max_text_length = max_text_length

        self._model = None
        self._chroma_client = None
        self._embedding_dim = None

        # Initialize based on provider
        if embedding_provider == "openai" and openai:
            if openai_api_key:
                openai.api_key = openai_api_key
            self._embedding_dim = 1536  # OpenAI ada-002 dimension
        elif embedding_provider == "ollama" and OllamaEmbeddings:
            self._load_ollama_embeddings()
        elif embedding_provider == "sentence_transformers" and SentenceTransformer:
            self._load_sentence_transformer()
        else:
            logger.warning(f"Embedding provider {embedding_provider} not available")

        # Initialize Chroma client
        self._initialize_chroma()

    def _load_sentence_transformer(self) -> None:
        """Load sentence transformer model."""
        try:
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer model {self.model_name} with dimension {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise EmbeddingError(f"Cannot load model {self.model_name}: {e}")

    def _load_ollama_embeddings(self) -> None:
        """Load Ollama embedding model."""
        try:
            self._model = OllamaEmbeddings(model=self.model_name)
            # embeddinggemma typically produces 2048-dimensional embeddings
            self._embedding_dim = 2048
            logger.info(f"Loaded Ollama embedding model {self.model_name} with dimension {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load Ollama embedding model: {e}")
            raise EmbeddingError(f"Cannot load model {self.model_name}: {e}")

    def _initialize_chroma(self) -> None:
        """Initialize Chroma vector database client."""
        if not chromadb:
            logger.warning("ChromaDB not available, vector storage disabled")
            return

        try:
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            logger.info(f"Initialized ChromaDB client at {self.chroma_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if generation fails
        """
        if not text or not text.strip():
            return None

        # Truncate text if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]

        try:
            if self.embedding_provider == "openai" and openai:
                return await self._generate_openai_embedding(text)
            elif self.embedding_provider == "ollama" and self._model:
                return await self._generate_ollama_embedding(text)
            elif self.embedding_provider == "sentence_transformers" and self._model:
                return await self._generate_sentence_transformer_embedding(text)
            else:
                logger.error("No embedding provider available")
                return None

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def _generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API."""
        try:
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            return None

    async def _generate_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.embed_query(text)
            )
            return embedding
        except Exception as e:
            logger.error(f"Ollama embedding generation failed: {e}")
            return None

    async def _generate_sentence_transformer_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using sentence transformers."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode([text], convert_to_tensor=False)[0]
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding generation failed: {e}")
            return None

    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (may contain None for failed embeddings)
        """
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._process_embedding_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _process_embedding_batch(
        self,
        batch: List[str]
    ) -> List[Optional[List[float]]]:
        """Process a single batch of texts for embedding."""
        if self.embedding_provider == "sentence_transformers" and self._model:
            try:
                # Truncate texts
                truncated_batch = [
                    text[:self.max_text_length] if len(text) > self.max_text_length else text
                    for text in batch
                ]

                # Generate embeddings in thread pool
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._model.encode(truncated_batch, convert_to_tensor=False)
                )

                return [emb.tolist() for emb in embeddings]

            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                return [None] * len(batch)

        elif self.embedding_provider == "ollama" and self._model:
            try:
                # Truncate texts
                truncated_batch = [
                    text[:self.max_text_length] if len(text) > self.max_text_length else text
                    for text in batch
                ]

                # Generate embeddings using Ollama
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._model.embed_documents(truncated_batch)
                )

                return embeddings

            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                return [None] * len(batch)

        else:
            # Fallback to individual processing for OpenAI or other providers
            embeddings = []
            for text in batch:
                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)
            return embeddings

    async def embed_code_entity(self, entity: CodeEntity) -> bool:
        """
        Generate and store embedding for a code entity.

        Args:
            entity: Code entity to embed

        Returns:
            True if embedding successful, False otherwise
        """
        # Create text representation for embedding
        text_parts = []

        # Add entity name and type
        entity_type_str = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        text_parts.append(f"{entity_type_str}: {entity.name}")

        # Add signature if available
        if entity.signature:
            text_parts.append(f"Signature: {entity.signature}")

        # Add docstring if available
        if entity.docstring:
            text_parts.append(f"Documentation: {entity.docstring}")

        # Add content snippet if available (truncated)
        if entity.content:
            content_snippet = entity.content[:500]  # First 500 chars
            text_parts.append(f"Content: {content_snippet}")

        # Create combined text
        text = "\n".join(text_parts)

        # Generate embedding
        embedding = await self.generate_embedding(text)

        if embedding:
            entity.embedding_vector = embedding
            entity.embedding_model = f"{self.embedding_provider}:{self.model_name}"
            return True

        return False

    async def embed_code_entities_batch(
        self,
        entities: List[CodeEntity]
    ) -> int:
        """
        Generate embeddings for multiple code entities.

        Args:
            entities: List of code entities

        Returns:
            Number of entities successfully embedded
        """
        if not entities:
            return 0

        # Prepare texts for embedding
        texts = []
        for entity in entities:
            text_parts = []
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
            text_parts.append(f"{entity_type}: {entity.name}")

            if entity.signature:
                text_parts.append(f"Signature: {entity.signature}")
            if entity.docstring:
                text_parts.append(f"Documentation: {entity.docstring}")
            if entity.content:
                content_snippet = entity.content[:500]
                text_parts.append(f"Content: {content_snippet}")

            texts.append("\n".join(text_parts))

        # Generate embeddings
        embeddings = await self.generate_embeddings_batch(texts)

        # Update entities with embeddings
        success_count = 0
        for entity, embedding in zip(entities, embeddings):
            if embedding:
                entity.embedding_vector = embedding
                entity.embedding_model = f"{self.embedding_provider}:{self.model_name}"
                success_count += 1

        return success_count

    async def store_embeddings_in_chroma(
        self,
        entities: List[CodeEntity],
        collection_name: str
    ) -> bool:
        """
        Store entity embeddings in Chroma vector database.

        Args:
            entities: Code entities with embeddings
            collection_name: Chroma collection name

        Returns:
            True if storage successful, False otherwise
        """
        if not self._chroma_client or not entities:
            return False

        try:
            # Get or create collection
            collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"embedding_model": f"{self.embedding_provider}:{self.model_name}"}
            )

            # Prepare data for storage
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for entity in entities:
                if entity.embedding_vector:
                    ids.append(entity.id)
                    embeddings.append(entity.embedding_vector)

                    # Create metadata
                    metadata = {
                        "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                        "name": entity.name,
                        "qualified_name": entity.qualified_name,
                        "file_path": entity.file_path,
                        "language": entity.language,
                        "repository_id": entity.repository_id,
                        "start_line": entity.start_line,
                        "end_line": entity.end_line,
                    }
                    metadatas.append(metadata)

                    # Create document text
                    entity_type_str = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                    doc_parts = [f"{entity_type_str}: {entity.name}"]
                    if entity.signature:
                        doc_parts.append(entity.signature)
                    if entity.docstring:
                        doc_parts.append(entity.docstring)
                    documents.append("\n".join(doc_parts))

            # Store in batches
            batch_size = 1000
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]

                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )

            logger.info(f"Stored {len(ids)} embeddings in collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to store embeddings in Chroma: {e}")
            return False

    async def similarity_search(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entities using vector similarity.

        Args:
            query_text: Text to search for
            collection_name: Chroma collection name
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of similar entities with scores
        """
        if not self._chroma_client:
            return []

        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query_text)
            if not query_embedding:
                return []

            # Get collection
            collection = self._chroma_client.get_collection(collection_name)

            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )

            # Format results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, entity_id in enumerate(results['ids'][0]):
                    result = {
                        "entity_id": entity_id,
                        "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 1.0,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "document": results['documents'][0][i] if results['documents'] else ""
                    }
                    search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    async def embed_query(self, query: Query) -> bool:
        """
        Generate embedding for a query.

        Args:
            query: Query object to embed

        Returns:
            True if embedding successful, False otherwise
        """
        embedding = await self.generate_embedding(query.query_text)

        if embedding:
            query.query_embedding = embedding
            query.embedding_model = f"{self.embedding_provider}:{self.model_name}"
            return True

        return False

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0

        try:
            if cos_sim and SentenceTransformer:
                import torch
                tensor1 = torch.tensor(embedding1).unsqueeze(0)
                tensor2 = torch.tensor(embedding2).unsqueeze(0)
                return float(cos_sim(tensor1, tensor2).item())
            else:
                # Manual cosine similarity calculation
                import math
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                magnitude_a = math.sqrt(sum(a * a for a in embedding1))
                magnitude_b = math.sqrt(sum(b * b for b in embedding2))

                if magnitude_a == 0 or magnitude_b == 0:
                    return 0.0

                return dot_product / (magnitude_a * magnitude_b)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings produced by this service."""
        return self._embedding_dim

    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return (
            (self.embedding_provider == "sentence_transformers" and self._model is not None) or
            (self.embedding_provider == "openai" and openai is not None)
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._chroma_client:
            # Chroma client cleanup is automatic
            pass

        if self._model:
            # SentenceTransformer cleanup
            del self._model
            self._model = None