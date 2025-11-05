# Phase 3: Advanced Features & Query System Implementation Guide
*Weeks 6-8: Multi-Model Embeddings, Query Intelligence, and Context Reconstruction*

## Overview

Phase 3 transforms the system from a document processor into a sophisticated semantic search engine. This phase implements multi-model embeddings, intelligent query routing, context reconstruction, and advanced search capabilities including comment thread processing and feedback loops.

## Technical Architecture Expansion

### Enhanced Project Structure
```
semantic_processor/
├── src/
│   ├── embeddings/                    # NEW
│   │   ├── __init__.py
│   │   ├── multi_model_embedder.py
│   │   ├── embedding_manager.py
│   │   └── embedding_optimizer.py
│   ├── search/                        # NEW
│   │   ├── __init__.py
│   │   ├── query_classifier.py
│   │   ├── search_engine.py
│   │   ├── result_ranker.py
│   │   └── context_reconstructor.py
│   ├── feedback/                      # NEW
│   │   ├── __init__.py
│   │   ├── feedback_collector.py
│   │   ├── model_updater.py
│   │   └── performance_tracker.py
│   ├── parsers/
│   │   └── comment_thread_parser.py   # Implementation from Phase 2
│   └── api/                          # NEW
│       ├── __init__.py
│       ├── search_api.py
│       └── admin_api.py
```

## Core Embedding System

### 1. Multi-Model Embedding Manager (`src/embeddings/multi_model_embedder.py`)

```python
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path
import time

@dataclass
class EmbeddingConfig:
    """Configuration for specific embedding model"""
    model_name: str
    model_path: str
    chunk_levels: List[str]  # Which chunk levels this model handles
    max_sequence_length: int
    embedding_dimension: int
    batch_size: int

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embeddings: np.ndarray
    model_used: str
    chunk_ids: List[str]
    processing_time: float
    batch_size: int
    average_confidence: float

class MultiModelEmbedder:
    """Manages multiple embedding models for different chunk types"""

    def __init__(self, config_path: Optional[Path] = None):
        self.models: Dict[str, SentenceTransformer] = {}
        self.model_configs: Dict[str, EmbeddingConfig] = {}
        self.performance_cache: Dict[str, Dict[str, float]] = {}

        # Load configuration
        if config_path:
            self._load_config(config_path)
        else:
            self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize default embedding models for different chunk levels"""
        default_configs = [
            EmbeddingConfig(
                model_name="concept_embedder",
                model_path="all-mpnet-base-v2",
                chunk_levels=["concept", "document"],
                max_sequence_length=512,
                embedding_dimension=768,
                batch_size=16
            ),
            EmbeddingConfig(
                model_name="section_embedder",
                model_path="all-MiniLM-L12-v2",
                chunk_levels=["relationship", "section"],
                max_sequence_length=512,
                embedding_dimension=384,
                batch_size=32
            ),
            EmbeddingConfig(
                model_name="content_embedder",
                model_path="all-MiniLM-L6-v2",
                chunk_levels=["content_block", "paragraph"],
                max_sequence_length=512,
                embedding_dimension=384,
                batch_size=64
            ),
            EmbeddingConfig(
                model_name="sentence_embedder",
                model_path="all-MiniLM-L6-v2",
                chunk_levels=["sentence"],
                max_sequence_length=256,
                embedding_dimension=384,
                batch_size=128
            ),
            EmbeddingConfig(
                model_name="academic_embedder",
                model_path="allenai-specter",  # Specialized for academic content
                chunk_levels=["academic_section", "academic_paragraph"],
                max_sequence_length=512,
                embedding_dimension=768,
                batch_size=16
            )
        ]

        for config in default_configs:
            self._load_model(config)

    def _load_model(self, config: EmbeddingConfig):
        """Load embedding model with configuration"""
        try:
            model = SentenceTransformer(config.model_path)
            model.max_seq_length = config.max_sequence_length

            self.models[config.model_name] = model
            self.model_configs[config.model_name] = config

            print(f"Loaded model: {config.model_name} ({config.model_path})")

        except Exception as e:
            print(f"Failed to load model {config.model_name}: {e}")

    def embed_chunks(
        self,
        chunks: List,
        force_model: Optional[str] = None
    ) -> Dict[str, EmbeddingResult]:
        """Generate embeddings for chunks using appropriate models"""

        # Group chunks by appropriate model
        model_chunks: Dict[str, List] = {}

        for chunk in chunks:
            model_name = force_model or self._select_model_for_chunk(chunk)
            if model_name not in model_chunks:
                model_chunks[model_name] = []
            model_chunks[model_name].append(chunk)

        # Generate embeddings for each model group
        results = {}
        for model_name, chunk_group in model_chunks.items():
            if model_name in self.models:
                result = self._embed_with_model(model_name, chunk_group)
                results[model_name] = result

        return results

    def _select_model_for_chunk(self, chunk) -> str:
        """Select optimal model for chunk based on level and content type"""
        chunk_level = chunk.level
        content_type = chunk.metadata.get('document_type', 'general')

        # Academic content gets specialized model
        if content_type == 'pdf' and 'academic' in chunk.metadata.get('title', '').lower():
            for model_name, config in self.model_configs.items():
                if 'academic' in model_name and chunk_level in config.chunk_levels:
                    return model_name

        # Standard model selection by chunk level
        for model_name, config in self.model_configs.items():
            if chunk_level in config.chunk_levels and 'academic' not in model_name:
                return model_name

        # Fallback to content embedder
        return "content_embedder"

    def _embed_with_model(
        self,
        model_name: str,
        chunks: List
    ) -> EmbeddingResult:
        """Generate embeddings using specific model"""

        model = self.models[model_name]
        config = self.model_configs[model_name]

        # Prepare texts for embedding
        texts = []
        chunk_ids = []

        for chunk in chunks:
            # Enhance text with metadata context for better embeddings
            enhanced_text = self._enhance_text_for_embedding(chunk)
            texts.append(enhanced_text)
            chunk_ids.append(chunk.id)

        # Generate embeddings in batches
        start_time = time.time()
        embeddings = model.encode(
            texts,
            batch_size=config.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        processing_time = time.time() - start_time

        # Calculate average confidence (simplified metric)
        avg_confidence = np.mean([chunk.confidence_score for chunk in chunks])

        return EmbeddingResult(
            embeddings=embeddings,
            model_used=model_name,
            chunk_ids=chunk_ids,
            processing_time=processing_time,
            batch_size=len(texts),
            average_confidence=avg_confidence
        )

    def _enhance_text_for_embedding(self, chunk) -> str:
        """Enhance chunk text with metadata for better embeddings"""
        enhanced_parts = [chunk.content]

        # Add context from metadata
        if chunk.metadata.get('concept_title'):
            enhanced_parts.insert(0, f"Concept: {chunk.metadata['concept_title']}")

        if chunk.metadata.get('relationship_type'):
            enhanced_parts.insert(0, f"Type: {chunk.metadata['relationship_type']}")

        if chunk.metadata.get('hierarchy_path'):
            enhanced_parts.insert(0, f"Section: {chunk.metadata['hierarchy_path']}")

        return ' | '.join(enhanced_parts)

class EmbeddingOptimizer:
    """Optimizes embedding models based on query performance"""

    def __init__(self, embedder: MultiModelEmbedder):
        self.embedder = embedder
        self.query_performance: Dict[str, List[float]] = {}
        self.model_usage_stats: Dict[str, int] = {}

    def record_query_performance(
        self,
        query: str,
        model_used: str,
        relevance_score: float,
        response_time: float
    ):
        """Record performance metrics for model optimization"""

        if model_used not in self.query_performance:
            self.query_performance[model_used] = []

        # Combined performance score (relevance weighted higher than speed)
        performance_score = (relevance_score * 0.7) + ((1.0 / max(response_time, 0.1)) * 0.3)
        self.query_performance[model_used].append(performance_score)

        self.model_usage_stats[model_used] = self.model_usage_stats.get(model_used, 0) + 1

    def get_model_recommendations(self) -> Dict[str, float]:
        """Get performance-based model recommendations"""
        recommendations = {}

        for model_name, performances in self.query_performance.items():
            if len(performances) >= 5:  # Minimum samples for reliable metrics
                avg_performance = np.mean(performances)
                usage_weight = min(1.0, self.model_usage_stats.get(model_name, 0) / 100)

                recommendations[model_name] = avg_performance * usage_weight

        return recommendations
```

**Implementation Questions for Phase 3:**

22. **Model Selection Strategy**: Should model selection be purely performance-based or should it also consider computational costs?

23. **Embedding Enhancement**: What metadata should be included in embedding text? Should it vary by chunk level?

24. **Model Update Frequency**: How often should embedding models be updated based on performance feedback?

### 2. Query Classification and Intelligence (`src/search/query_classifier.py`)

```python
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from pathlib import Path

class QueryType(Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"
    RELATIONSHIP = "relationship"

@dataclass
class QueryAnalysis:
    """Analysis result for a query"""
    original_query: str
    query_type: QueryType
    confidence: float
    key_terms: List[str]
    entities: List[str]
    suggested_chunk_levels: List[str]
    search_strategy: str
    expanded_query: Optional[str] = None

class QueryClassifier:
    """Intelligent query classification and analysis"""

    def __init__(self, model_path: Optional[Path] = None):
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier: Optional[MultinomialNB] = None
        self.vectorizer: Optional[TfidfVectorizer] = None

        # Query pattern definitions
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what is|what are|define|definition of|meaning of)\b',
                r'\b(how much|how many|when|where|who)\b',
                r'\b(temperature|pressure|value|amount|number)\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(difference|compare|contrast|versus|vs|better|worse)\b',
                r'\b(similarities|different from|compared to)\b',
                r'\b(pros and cons|advantages|disadvantages)\b'
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|how do|steps|process|procedure|configure)\b',
                r'\b(install|setup|implement|execute|run)\b',
                r'\b(guide|tutorial|instructions|method)\b'
            ],
            QueryType.RELATIONSHIP: [
                r'\b(related to|connection|relationship|link|association)\b',
                r'\b(causes|leads to|results in|affects|influences)\b',
                r'\b(depends on|requires|needs|prerequisite)\b'
            ]
        }

        # Load or initialize classifier
        if model_path and model_path.exists():
            self._load_classifier(model_path)
        else:
            self._initialize_rule_based_classifier()

    def _initialize_rule_based_classifier(self):
        """Initialize rule-based classifier as fallback"""
        pass

    def _load_classifier(self, model_path: Path):
        """Load trained classifier model"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.classifier = data['classifier']
                self.vectorizer = data['vectorizer']
        except Exception as e:
            print(f"Failed to load classifier: {e}")
            self._initialize_rule_based_classifier()

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis"""

        # Basic preprocessing
        query = query.strip().lower()
        doc = self.nlp(query)

        # Classify query type
        query_type, confidence = self._classify_query_type(query, doc)

        # Extract key terms and entities
        key_terms = self._extract_key_terms(doc)
        entities = self._extract_entities(doc)

        # Determine optimal search strategy
        chunk_levels, search_strategy = self._determine_search_strategy(query_type, query)

        # Expand query if beneficial
        expanded_query = self._expand_query(query, entities, key_terms)

        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            confidence=confidence,
            key_terms=key_terms,
            entities=entities,
            suggested_chunk_levels=chunk_levels,
            search_strategy=search_strategy,
            expanded_query=expanded_query
        )

    def _classify_query_type(self, query: str, doc) -> Tuple[QueryType, float]:
        """Classify query type using patterns and ML"""

        # Rule-based classification
        pattern_scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query))
                score += matches
            pattern_scores[query_type] = score

        # Find highest scoring type
        if pattern_scores:
            best_type = max(pattern_scores, key=pattern_scores.get)
            max_score = pattern_scores[best_type]

            if max_score > 0:
                # Calculate confidence based on score and pattern specificity
                confidence = min(0.9, 0.6 + (max_score * 0.1))
                return best_type, confidence

        # Fallback to linguistic analysis
        return self._linguistic_classification(doc)

    def _linguistic_classification(self, doc) -> Tuple[QueryType, float]:
        """Classify using linguistic features"""

        # Analyze POS tags and dependency relations
        has_wh_word = any(token.tag_ in ['WP', 'WRB', 'WDT'] for token in doc)
        has_comparative = any(token.tag_ in ['JJR', 'RBR'] for token in doc)
        has_modal = any(token.lemma_ in ['can', 'could', 'should', 'would', 'how'] for token in doc)

        if has_comparative:
            return QueryType.COMPARATIVE, 0.7
        elif has_modal and 'how' in [token.text.lower() for token in doc]:
            return QueryType.PROCEDURAL, 0.7
        elif has_wh_word:
            return QueryType.FACTUAL, 0.6
        else:
            return QueryType.CONCEPTUAL, 0.5

    def _extract_key_terms(self, doc) -> List[str]:
        """Extract key terms from query"""
        key_terms = []
        for token in doc:
            # Extract nouns, verbs, and adjectives that aren't stop words
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
                key_terms.append(token.lemma_)
        return key_terms

    def _extract_entities(self, doc) -> List[str]:
        """Extract named entities from query"""
        return [ent.text for ent in doc.ents]

    def _determine_search_strategy(
        self,
        query_type: QueryType,
        query: str
    ) -> Tuple[List[str], str]:
        """Determine optimal chunk levels and search strategy"""

        strategies = {
            QueryType.FACTUAL: (
                ["sentence", "content_block"],
                "precise_search"
            ),
            QueryType.CONCEPTUAL: (
                ["content_block", "relationship", "concept"],
                "comprehensive_search"
            ),
            QueryType.COMPARATIVE: (
                ["relationship", "content_block", "concept"],
                "multi_document_search"
            ),
            QueryType.PROCEDURAL: (
                ["content_block", "section"],
                "sequential_search"
            ),
            QueryType.RELATIONSHIP: (
                ["relationship", "concept"],
                "graph_aware_search"
            ),
            QueryType.EXPLORATORY: (
                ["concept", "relationship", "content_block"],
                "broad_search"
            )
        }

        return strategies.get(query_type, (["content_block"], "standard_search"))

    def _expand_query(
        self,
        query: str,
        entities: List[str],
        key_terms: List[str]
    ) -> Optional[str]:
        """Expand query with related terms and synonyms"""

        # Simple expansion based on entities and key terms
        expansion_terms = []

        # Add related technical terms (this could be enhanced with a knowledge base)
        tech_expansions = {
            'machine learning': ['ML', 'artificial intelligence', 'neural networks'],
            'temperature': ['thermal', 'heat', 'cooling'],
            'pressure': ['force', 'compression', 'stress'],
        }

        for term in key_terms:
            if term in tech_expansions:
                expansion_terms.extend(tech_expansions[term])

        if expansion_terms:
            return f"{query} ({' OR '.join(expansion_terms)})"

        return None

class QueryRouter:
    """Routes queries to appropriate search strategies"""

    def __init__(self, classifier: QueryClassifier):
        self.classifier = classifier
        self.strategy_handlers = {
            "precise_search": self._handle_precise_search,
            "comprehensive_search": self._handle_comprehensive_search,
            "multi_document_search": self._handle_multi_document_search,
            "sequential_search": self._handle_sequential_search,
            "graph_aware_search": self._handle_graph_aware_search,
            "broad_search": self._handle_broad_search
        }

    def route_query(
        self,
        query: str,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Route query to appropriate search strategy"""

        analysis = self.classifier.analyze_query(query)

        # Get strategy handler
        handler = self.strategy_handlers.get(
            analysis.search_strategy,
            self._handle_standard_search
        )

        # Execute search strategy
        results = handler(analysis, search_engine)

        # Add analysis metadata to results
        results['query_analysis'] = analysis

        return results

    def _handle_precise_search(
        self,
        analysis: QueryAnalysis,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Handle precise fact-finding queries"""

        query_text = analysis.expanded_query or analysis.original_query

        # Search sentence and content_block levels with high precision
        results = search_engine.search_multi_level(
            query=query_text,
            levels=["sentence", "content_block"],
            n_results_per_level=5,
            similarity_threshold=0.8,
            rerank_results=True
        )

        return {
            'search_type': 'precise',
            'results': results,
            'confidence': analysis.confidence
        }

    def _handle_comprehensive_search(
        self,
        analysis: QueryAnalysis,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Handle comprehensive conceptual queries"""

        query_text = analysis.expanded_query or analysis.original_query

        results = search_engine.search_multi_level(
            query=query_text,
            levels=["content_block", "relationship", "concept"],
            n_results_per_level=8,
            similarity_threshold=0.6,
            rerank_results=True
        )

        return {
            'search_type': 'comprehensive',
            'results': results,
            'confidence': analysis.confidence
        }

    def _handle_multi_document_search(
        self,
        analysis: QueryAnalysis,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Handle comparative queries across documents"""

        query_text = analysis.expanded_query or analysis.original_query

        results = search_engine.search_multi_level(
            query=query_text,
            levels=["relationship", "content_block", "concept"],
            n_results_per_level=10,
            similarity_threshold=0.65,
            rerank_results=True
        )

        return {
            'search_type': 'multi_document',
            'results': results,
            'confidence': analysis.confidence
        }

    def _handle_sequential_search(
        self,
        analysis: QueryAnalysis,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Handle procedural/sequential queries"""

        query_text = analysis.expanded_query or analysis.original_query

        results = search_engine.search_multi_level(
            query=query_text,
            levels=["content_block", "section"],
            n_results_per_level=6,
            similarity_threshold=0.7,
            rerank_results=True
        )

        return {
            'search_type': 'sequential',
            'results': results,
            'confidence': analysis.confidence
        }

    def _handle_graph_aware_search(
        self,
        analysis: QueryAnalysis,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Handle relationship-focused queries using knowledge graph"""

        # Find related concepts first
        initial_results = search_engine.search_concepts(
            query=analysis.original_query,
            n_results=10
        )

        # Expand search using org-roam links and references
        expanded_results = search_engine.expand_with_relationships(
            initial_results,
            max_depth=2
        )

        return {
            'search_type': 'graph_aware',
            'results': expanded_results,
            'relationship_depth': 2,
            'confidence': analysis.confidence
        }

    def _handle_broad_search(
        self,
        analysis: QueryAnalysis,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Handle exploratory broad searches"""

        query_text = analysis.expanded_query or analysis.original_query

        results = search_engine.search_multi_level(
            query=query_text,
            levels=["concept", "relationship", "content_block"],
            n_results_per_level=15,
            similarity_threshold=0.55,
            rerank_results=True
        )

        return {
            'search_type': 'broad',
            'results': results,
            'confidence': analysis.confidence
        }

    def _handle_standard_search(
        self,
        analysis: QueryAnalysis,
        search_engine: 'SearchEngine'
    ) -> Dict[str, Any]:
        """Handle standard search as fallback"""

        query_text = analysis.expanded_query or analysis.original_query

        results = search_engine.search_multi_level(
            query=query_text,
            levels=["content_block"],
            n_results_per_level=10,
            similarity_threshold=0.65,
            rerank_results=False
        )

        return {
            'search_type': 'standard',
            'results': results,
            'confidence': analysis.confidence
        }
```

### 3. Advanced Search Engine (`src/search/search_engine.py`)

```python
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class SearchResult:
    """Individual search result"""
    chunk_id: str
    content: str
    similarity_score: float
    chunk_level: str
    metadata: Dict[str, Any]
    context_path: str
    related_chunks: List[str] = None

@dataclass
class SearchResponse:
    """Complete search response"""
    results: List[SearchResult]
    total_results: int
    query_analysis: Any  # QueryAnalysis type
    search_time: float
    context_reconstructed: bool = False

class AdvancedSearchEngine:
    """Advanced semantic search with multi-level querying"""

    def __init__(
        self,
        chroma_client,
        embedder,
        context_reconstructor: 'ContextReconstructor'
    ):
        self.chroma_client = chroma_client
        self.embedder = embedder
        self.context_reconstructor = context_reconstructor
        self.query_history: List[Dict[str, Any]] = []

    def search_multi_level(
        self,
        query: str,
        levels: List[str] = None,
        n_results_per_level: int = 10,
        similarity_threshold: float = 0.7,
        rerank_results: bool = True,
        include_context: bool = True
    ) -> SearchResponse:
        """Multi-level semantic search"""

        start_time = time.time()

        # Generate query embedding using appropriate model
        query_embedding = self._generate_query_embedding(query, levels)

        # Search across specified levels
        level_results = {}
        for level in levels or ["content_block", "relationship", "concept"]:
            collection_name = self._level_to_collection(level)
            if collection_name in self.chroma_client.collections:
                results = self._search_collection(
                    collection_name,
                    query_embedding,
                    n_results_per_level,
                    similarity_threshold
                )
                level_results[level] = results

        # Combine and rank results
        combined_results = self._combine_level_results(level_results)

        # Re-rank if requested
        if rerank_results:
            combined_results = self._rerank_results(query, combined_results)

        # Add context if requested
        if include_context:
            combined_results = self._add_context_to_results(combined_results)

        search_time = time.time() - start_time

        # Record query for learning
        self._record_query(query, combined_results, search_time)

        return SearchResponse(
            results=combined_results,
            total_results=len(combined_results),
            query_analysis=None,  # Will be added by QueryRouter
            search_time=search_time,
            context_reconstructed=include_context
        )

    def _generate_query_embedding(self, query: str, levels: Optional[List[str]]) -> np.ndarray:
        """Generate embedding for query"""
        # Use first level's model or default to content embedder
        if levels and len(levels) > 0:
            model_name = self._select_model_for_level(levels[0])
        else:
            model_name = "content_embedder"

        if model_name in self.embedder.models:
            model = self.embedder.models[model_name]
            embedding = model.encode([query], convert_to_numpy=True)
            return embedding[0]

        # Fallback
        return np.zeros(384)

    def _select_model_for_level(self, level: str) -> str:
        """Select appropriate model for chunk level"""
        level_model_map = {
            'concept': 'concept_embedder',
            'document': 'concept_embedder',
            'relationship': 'section_embedder',
            'section': 'section_embedder',
            'content_block': 'content_embedder',
            'paragraph': 'content_embedder',
            'sentence': 'sentence_embedder'
        }
        return level_model_map.get(level, 'content_embedder')

    def _level_to_collection(self, level: str) -> str:
        """Map chunk level to collection name"""
        level_map = {
            'concept': 'concepts',
            'document': 'concepts',
            'relationship': 'relationships',
            'section': 'relationships',
            'content_block': 'content_blocks',
            'paragraph': 'content_blocks',
            'sentence': 'sentences'
        }
        return level_map.get(level, 'content_blocks')

    def _search_collection(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        n_results: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search specific collection"""
        collection = self.chroma_client.collections[collection_name]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Convert to SearchResult objects
        search_results = []
        if results and results['ids'] and len(results['ids']) > 0:
            for i, chunk_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                if similarity >= similarity_threshold:
                    search_results.append({
                        'chunk_id': chunk_id,
                        'content': results['documents'][0][i],
                        'similarity_score': similarity,
                        'chunk_level': collection_name,
                        'metadata': results['metadatas'][0][i],
                        'context_path': results['metadatas'][0][i].get('hierarchy_path', '')
                    })

        return search_results

    def _combine_level_results(self, level_results: Dict[str, List[Dict[str, Any]]]) -> List[SearchResult]:
        """Combine results from different levels"""
        combined = []

        for level, results in level_results.items():
            for result in results:
                search_result = SearchResult(
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    similarity_score=result['similarity_score'],
                    chunk_level=result['chunk_level'],
                    metadata=result['metadata'],
                    context_path=result['context_path']
                )
                combined.append(search_result)

        # Sort by similarity score
        combined.sort(key=lambda r: r.similarity_score, reverse=True)
        return combined

    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Advanced re-ranking using multiple signals"""

        for result in results:
            # Base score from similarity
            score = result.similarity_score

            # Boost for complete concepts
            if result.chunk_level == 'concepts':
                score *= 1.2

            # Boost for high-confidence chunks
            confidence = result.metadata.get('confidence_score', 0.5)
            score *= (0.8 + confidence * 0.4)

            # Boost for documents with many connections
            link_count = len(result.metadata.get('org_roam_links', []))
            if link_count > 0:
                score *= (1.0 + min(0.3, link_count * 0.05))

            result.similarity_score = score

        return sorted(results, key=lambda r: r.similarity_score, reverse=True)

    def _add_context_to_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Add hierarchical context to results"""
        # Context reconstruction would happen here
        # For now, just return results as-is
        return results

    def _record_query(self, query: str, results: List[SearchResult], search_time: float):
        """Record query for analysis"""
        self.query_history.append({
            'query': query,
            'result_count': len(results),
            'search_time': search_time,
            'timestamp': time.time()
        })

    def search_concepts(self, query: str, n_results: int = 10) -> List[SearchResult]:
        """Search only concept-level chunks"""
        return self.search_multi_level(
            query=query,
            levels=["concept"],
            n_results_per_level=n_results,
            include_context=False
        ).results

    def expand_with_relationships(
        self,
        initial_results: List[SearchResult],
        max_depth: int = 2
    ) -> List[SearchResult]:
        """Expand results using document relationships"""

        expanded_results = list(initial_results)
        processed_docs = set()

        for depth in range(max_depth):
            current_batch = []

            for result in expanded_results:
                doc_id = result.metadata.get('concept_title') or result.metadata.get('file_path')

                if doc_id and doc_id not in processed_docs:
                    # Find related documents
                    related_docs = self._find_related_documents(result)
                    current_batch.extend(related_docs)
                    processed_docs.add(doc_id)

            expanded_results.extend(current_batch)

        # Remove duplicates and re-rank
        unique_results = self._deduplicate_results(expanded_results)
        return self._limit_and_rank_results(unique_results, 50)

    def _find_related_documents(self, result: SearchResult) -> List[SearchResult]:
        """Find documents related through org-roam links and references"""
        related = []

        # Get org-roam links
        org_links = result.metadata.get('org_roam_links', [])
        if isinstance(org_links, str):
            import json
            try:
                org_links = json.loads(org_links)
            except:
                org_links = []

        for link in org_links[:5]:  # Limit to prevent explosion
            # Search for chunks from linked documents
            linked_results = self._search_by_document(link)
            related.extend(linked_results[:2])  # Top 2 from each linked doc

        # Get bibliography references
        bib_refs = result.metadata.get('bibliography_refs', [])
        if isinstance(bib_refs, str):
            import json
            try:
                bib_refs = json.loads(bib_refs)
            except:
                bib_refs = []

        if bib_refs:
            # Find other chunks with same references
            ref_results = self._search_by_references(bib_refs)
            related.extend(ref_results[:3])

        return related

    def _search_by_document(self, doc_identifier: str) -> List[SearchResult]:
        """Search for chunks from a specific document"""
        # Placeholder implementation
        return []

    def _search_by_references(self, references: List[str]) -> List[SearchResult]:
        """Search for chunks with specific bibliography references"""
        # Placeholder implementation
        return []

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results"""
        seen_ids = set()
        unique = []

        for result in results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                unique.append(result)

        return unique

    def _limit_and_rank_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        """Limit and rank results"""
        sorted_results = sorted(results, key=lambda r: r.similarity_score, reverse=True)
        return sorted_results[:limit]
```

### 4. Context Reconstruction System (`src/search/context_reconstructor.py`)

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ContextualResult:
    """Search result with reconstructed context"""
    base_result: SearchResult
    hierarchical_context: Dict[str, str]  # level -> content
    related_chunks: List[SearchResult]
    context_confidence: float

class ContextReconstructor:
    """Reconstructs hierarchical context for search results"""

    def __init__(self, chroma_client):
        self.chroma_client = chroma_client

    def reconstruct_context(
        self,
        results: List[SearchResult],
        max_context_levels: int = 3
    ) -> List[ContextualResult]:
        """Reconstruct hierarchical context for search results"""

        contextual_results = []

        for result in results:
            # Get hierarchical context
            context = self._build_hierarchical_context(result, max_context_levels)

            # Get related chunks at same level
            related = self._get_related_chunks(result)

            # Calculate context confidence
            confidence = self._calculate_context_confidence(context, related)

            contextual_results.append(ContextualResult(
                base_result=result,
                hierarchical_context=context,
                related_chunks=related,
                context_confidence=confidence
            ))

        return contextual_results

    def _build_hierarchical_context(
        self,
        result: SearchResult,
        max_levels: int
    ) -> Dict[str, str]:
        """Build context hierarchy for a result"""

        context = {}
        current_chunk_id = result.chunk_id

        # Walk up the hierarchy
        for level in range(max_levels):
            parent_info = self._get_parent_chunk(current_chunk_id, result.chunk_level)
            if not parent_info:
                break

            parent_chunk, parent_level = parent_info
            context[parent_level] = parent_chunk['content'][:300]  # Truncate for brevity
            current_chunk_id = parent_chunk['metadata'].get('parent_id')

            if not current_chunk_id:
                break

        return context

    def _get_parent_chunk(self, chunk_id: str, current_level: str) -> Optional[Tuple[Dict[str, Any], str]]:
        """Get parent chunk from ChromaDB"""
        # Get current chunk to find parent_id
        current_collection = self._level_to_collection(current_level)
        if current_collection not in self.chroma_client.collections:
            return None

        collection = self.chroma_client.collections[current_collection]

        try:
            result = collection.get(ids=[chunk_id])
            if not result or not result['metadatas']:
                return None

            metadata = result['metadatas'][0]
            parent_id = metadata.get('parent_id')

            if not parent_id:
                return None

            # Infer parent collection
            parent_level = self._infer_parent_level(current_level)
            if not parent_level:
                return None

            parent_collection = self._level_to_collection(parent_level)
            if parent_collection not in self.chroma_client.collections:
                return None

            # Get parent chunk
            parent_coll = self.chroma_client.collections[parent_collection]
            parent_result = parent_coll.get(ids=[parent_id])

            if parent_result and parent_result['documents']:
                return {
                    'content': parent_result['documents'][0],
                    'metadata': parent_result['metadatas'][0]
                }, parent_level

        except Exception as e:
            print(f"Error getting parent chunk: {e}")

        return None

    def _infer_parent_level(self, current_level: str) -> Optional[str]:
        """Infer parent level from current level"""
        hierarchy = {
            'sentences': 'content_blocks',
            'content_blocks': 'relationships',
            'relationships': 'concepts'
        }
        return hierarchy.get(current_level)

    def _level_to_collection(self, level: str) -> str:
        """Map level to collection name"""
        level_map = {
            'concept': 'concepts',
            'relationship': 'relationships',
            'content_block': 'content_blocks',
            'sentence': 'sentences',
            'concepts': 'concepts',
            'relationships': 'relationships',
            'content_blocks': 'content_blocks',
            'sentences': 'sentences'
        }
        return level_map.get(level, 'content_blocks')

    def _get_related_chunks(self, result: SearchResult) -> List[SearchResult]:
        """Get sibling chunks at same level"""
        # Placeholder - would retrieve sibling chunks
        return []

    def _calculate_context_confidence(
        self,
        context: Dict[str, str],
        related: List[SearchResult]
    ) -> float:
        """Calculate confidence in context reconstruction"""
        base_confidence = 0.7

        # Boost for multiple context levels
        if len(context) >= 2:
            base_confidence += 0.15

        # Boost for related chunks
        if related:
            base_confidence += min(0.15, len(related) * 0.05)

        return min(1.0, base_confidence)

    def format_contextual_result(
        self,
        contextual_result: ContextualResult,
        include_siblings: bool = True
    ) -> str:
        """Format contextual result for display"""

        result = contextual_result.base_result
        context = contextual_result.hierarchical_context

        # Build formatted output
        output_parts = []

        # Add document/concept context
        if 'concepts' in context:
            output_parts.append(f"📖 **Concept**: {context['concepts'][:100]}...")

        # Add section context
        if 'relationships' in context:
            output_parts.append(f"📑 **Section**: {context['relationships'][:100]}...")

        # Add main result
        output_parts.append(f"🎯 **Match**: {result.content}")

        # Add metadata
        metadata_parts = []
        if result.metadata.get('file_path'):
            file_name = Path(result.metadata['file_path']).name
            metadata_parts.append(f"File: {file_name}")

        if result.metadata.get('hierarchy_path'):
            metadata_parts.append(f"Location: {result.metadata['hierarchy_path']}")

        if metadata_parts:
            output_parts.append(f"ℹ️ **Source**: {' | '.join(metadata_parts)}")

        # Add related chunks if requested
        if include_siblings and contextual_result.related_chunks:
            related_preview = contextual_result.related_chunks[0].content[:100]
            output_parts.append(f"🔗 **Related**: {related_preview}...")

        return '\n\n'.join(output_parts)
```

## Phase 3 Deliverables

### Week 6: Embedding and Search Infrastructure
- [ ] Multi-model embedding system
- [ ] Query classification and routing
- [ ] Advanced search engine with multi-level queries
- [ ] Performance optimization for embeddings

### Week 7: Context and Intelligence Features
- [ ] Context reconstruction system
- [ ] Relationship-aware search
- [ ] Comment thread processing
- [ ] Query expansion and enhancement

### Week 8: Integration and Feedback Systems
- [ ] Complete search API
- [ ] Feedback collection and model updates
- [ ] Performance monitoring dashboard
- [ ] Documentation and testing

## Critical Decisions for Phase 3

25. **Embedding Model Performance vs Cost**: Should the system prioritize accuracy with larger models or speed with smaller models?

26. **Context Reconstruction Depth**: How many levels of context should be reconstructed by default?

27. **Query Learning**: Should the system automatically adapt to user query patterns, and how quickly should it adapt?

## Future Capabilities (Post Phase 3)

### Advanced Query Intelligence
- **Natural Language Query Processing**: Full conversational query interface
- **Query Intent Prediction**: Predict user intent before they finish typing
- **Multi-Modal Queries**: Support for image and voice queries
- **Query Suggestions**: Intelligent suggestions based on current search context

### Enhanced Context Awareness
- **Temporal Context**: Consider time-based relationships between documents
- **User Context**: Personalize results based on user's research history
- **Cross-Domain Context**: Link concepts across different knowledge domains
- **Dynamic Context**: Adapt context based on user interaction patterns

### Advanced Analytics
- **Knowledge Gap Detection**: Identify missing connections in knowledge base
- **Concept Evolution Tracking**: Track how concepts change over time
- **Research Trend Analysis**: Identify emerging patterns in document collection
- **Query Pattern Analysis**: Understand common research paths

### Performance and Scalability
- **Distributed Search**: Scale across multiple machines
- **Caching Strategies**: Intelligent caching of frequent queries
- **Incremental Index Updates**: Update search index without full rebuild
- **Query Result Streaming**: Stream results for very large result sets

## Related Documentation

For conceptual background on the techniques used in this implementation:

- **Query Classification**: [query-classification-routing.md](query-classification-routing.md)
- **Embedding Strategies**: [semantic-search-embedding-strategies.md](semantic-search-embedding-strategies.md)
- **Chunking Concepts**: [hierarchical-document-chunking-strategies.md](hierarchical-document-chunking-strategies.md)
- **Org-roam Integration**: [org-roam-zettelkasten-semantic-search.md](org-roam-zettelkasten-semantic-search.md)
- **ChromaDB Usage**: [chromadb-hierarchical-storage.md](chromadb-hierarchical-storage.md)

For project overview and previous phases:

- **Project Specification**: [semantic-document-processing-project-specification.md](semantic-document-processing-project-specification.md)
- **Phase 1 Implementation**: [semantic-document-processing-phase1-implementation.md](semantic-document-processing-phase1-implementation.md)
- **Phase 2 Implementation**: [semantic-document-processing-phase2-implementation.md](semantic-document-processing-phase2-implementation.md)

This Phase 3 implementation completes the semantic search system, transforming document processing into an intelligent, context-aware knowledge retrieval system with multi-model embeddings and advanced query understanding.
