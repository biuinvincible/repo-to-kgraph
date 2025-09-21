"""
Embedding service for generating semantic vectors.

Handles text embedding generation using HuggingFace CodeT5+ model via LangChain
with vector storage in Chroma database for similarity search.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

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

    Uses SentenceTransformer with Salesforce/SweRankEmbed-Small for code-optimized embeddings
    with optimized batch processing and vector storage in ChromaDB.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/SweRankEmbed-Small",
        chroma_db_path: str = "./chroma_db",
        batch_size: int = 128,
        max_text_length: int = 8192,
        device: str = "cpu"  # Use "cuda" for GPU acceleration if available
    ):
        """
        Initialize the embedding service with SentenceTransformer.

        Args:
            model_name: Model name (default: Salesforce/SweRankEmbed-Small for code ranking)
            chroma_db_path: Path to Chroma database storage
            batch_size: Batch size for processing
            max_text_length: Maximum text length for embeddings
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.chroma_db_path = chroma_db_path
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.device = device
        self.embedding_provider = "sentence_transformers"

        self._model = None
        self._chroma_client = None
        self._embedding_dim = None

        # Initialize SentenceTransformer model
        self._load_sentence_transformer()

        # Initialize Chroma client
        self._initialize_chroma()

    def _load_sentence_transformer(self) -> None:
        """Load SentenceTransformer model for code embeddings."""
        if not SentenceTransformer:
            raise EmbeddingError("sentence-transformers not available. Install with: pip install sentence-transformers")

        try:
            logger.info(f"Loading SentenceTransformer model {self.model_name}...")

            # Initialize SentenceTransformer with trust_remote_code=True for SweRankEmbed
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )

            # Test embedding to get dimension
            test_embedding = self._model.encode(["test"])
            self._embedding_dim = len(test_embedding[0])

            logger.info(f"Loaded SentenceTransformer model {self.model_name}")
            logger.info(f"Embedding dimension: {self._embedding_dim}")
            logger.info(f"Device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
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
        Generate embedding for a single text using HuggingFace CodeBERT.

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
            return await self._generate_sentence_transformer_embedding(text)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def _generate_sentence_transformer_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using SentenceTransformer model."""
        if not self._model:
            logger.error("SentenceTransformer model not loaded")
            return None

        try:
            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            # Use SentenceTransformer.encode() for document embeddings
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode([text])[0].tolist()
            )
            return embedding
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
        """Process a single batch of texts for embedding using SentenceTransformer."""
        if not self._model:
            logger.error("SentenceTransformer model not loaded")
            return [None] * len(batch)

        try:
            # Truncate texts to max length
            truncated_batch = [
                text[:self.max_text_length] if len(text) > self.max_text_length else text
                for text in batch
            ]

            # Generate embeddings in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(truncated_batch).tolist()
            )

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            # Fallback to individual processing
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

        # Add full content for better context (if available)
        if entity.content:
            text_parts.append(f"Implementation:\n{entity.content}")

        # Add semantic analysis information
        if entity.dependencies:
            text_parts.append(f"Dependencies: {', '.join(entity.dependencies)}")

        if entity.control_flow_info:
            flow_desc = self._describe_control_flow(entity.control_flow_info)
            if flow_desc:
                text_parts.append(f"Control Flow: {flow_desc}")

        if entity.error_handling:
            error_desc = self._describe_error_handling(entity.error_handling)
            if error_desc:
                text_parts.append(f"Error Handling: {error_desc}")

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
        embedding = await self.generate_query_embedding(query.query_text)

        if embedding:
            query.query_embedding = embedding
            query.embedding_model = f"{self.embedding_provider}:{self.model_name}"
            return True

        return False

    async def generate_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """
        Generate embedding for a query using SweRankEmbed with proper prompt.

        Args:
            query_text: Query text to embed

        Returns:
            Embedding vector or None if failed
        """
        if not self._model:
            logger.error("Model not loaded")
            return None

        try:
            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            # Use SentenceTransformer.encode() with prompt_name="query" for SweRankEmbed
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode([query_text], prompt_name="query")[0].tolist()
            )
            return embedding
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            # Fallback to regular embedding (without prompt_name)
            return await self.generate_embedding(query_text)

    def _describe_control_flow(self, control_flow_info: Dict[str, Any]) -> str:
        """Create human-readable description of control flow."""
        parts = []

        if control_flow_info.get("branches", 0) > 0:
            parts.append(f"{control_flow_info['branches']} conditional branches")

        if control_flow_info.get("loops", 0) > 0:
            parts.append(f"{control_flow_info['loops']} loops")

        if control_flow_info.get("returns", 0) > 1:
            parts.append(f"multiple return paths")

        complexity = control_flow_info.get("complexity_indicators", [])
        if complexity:
            parts.append(f"contains {', '.join(complexity[:3])}")

        return "; ".join(parts) if parts else ""

    def _describe_error_handling(self, error_handling: Dict[str, Any]) -> str:
        """Create human-readable description of error handling."""
        parts = []

        try_blocks = error_handling.get("try_blocks", 0)
        if try_blocks > 0:
            parts.append(f"{try_blocks} try-except blocks")

        exception_types = error_handling.get("exception_types", [])
        if exception_types:
            parts.append(f"handles {', '.join(exception_types[:3])}")

        if error_handling.get("has_finally"):
            parts.append("includes finally block")

        return "; ".join(parts) if parts else ""

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
            # Try to use torch/sentence-transformers for optimized calculation
            try:
                import torch
                from sentence_transformers.util import cos_sim
                tensor1 = torch.tensor(embedding1).unsqueeze(0)
                tensor2 = torch.tensor(embedding2).unsqueeze(0)
                return float(cos_sim(tensor1, tensor2).item())
            except ImportError:
                pass

            # Fallback to manual cosine similarity calculation
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