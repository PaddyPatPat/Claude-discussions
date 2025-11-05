# Semantic Search Embedding Strategies

## Overview

Embeddings are vector representations of text that capture semantic meaning. For hierarchical document chunking, different chunk sizes require different embedding strategies to optimize search quality across varied query types.

## Embedding Model Selection

### Popular Sentence-Transformer Models

**all-MiniLM-L6-v2** (Lightweight)
- Dimensions: 384
- Speed: Very fast (~14,000 sentences/second on GPU)
- Use case: Sentence-level chunks, high-throughput applications
- Quality: Good for general purposes

**all-MiniLM-L12-v2** (Balanced)
- Dimensions: 384
- Speed: Fast (~7,000 sentences/second on GPU)
- Use case: Paragraph-level chunks
- Quality: Better than L6, still efficient

**all-mpnet-base-v2** (High Quality)
- Dimensions: 768
- Speed: Moderate (~2,500 sentences/second on GPU)
- Use case: Section/chapter-level chunks, quality-critical applications
- Quality: One of the best general-purpose models

**multi-qa-mpnet-base-dot-v1** (Question-Answering Optimized)
- Dimensions: 768
- Speed: Moderate
- Use case: When queries are questions and documents are answers
- Quality: Excellent for Q&A retrieval

### Domain-Specific Models

**SciBERT**
- Domain: Scientific papers, academic content
- Use case: Research articles, technical documentation

**BioBERT**
- Domain: Biomedical and healthcare texts
- Use case: Medical documents, clinical notes

**Legal-BERT**
- Domain: Legal documents
- Use case: Contracts, case law, legal briefs

## Multi-Model Strategy for Hierarchical Chunks

### Approach: Different Models for Different Levels

```python
embedding_config = {
    "sentence": {
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "rationale": "Fast, precise retrieval for facts"
    },
    "paragraph": {
        "model": "all-MiniLM-L12-v2",
        "dimensions": 384,
        "rationale": "Balanced speed and quality for common queries"
    },
    "section": {
        "model": "all-mpnet-base-v2",
        "dimensions": 768,
        "rationale": "Higher quality for conceptual understanding"
    },
    "chapter": {
        "model": "all-mpnet-base-v2",
        "dimensions": 768,
        "rationale": "Captures broad themes and relationships"
    }
}
```

### Trade-offs

**Single Model (Simpler)**
- Pros: Easier to manage, consistent similarity scores across levels
- Cons: Not optimized for different chunk sizes

**Multi-Model (Optimized)**
- Pros: Better performance per level, lower costs for smaller chunks
- Cons: Cannot directly compare scores across levels, more complex implementation

**Recommendation**: Start with single model (all-mpnet-base-v2); migrate to multi-model if performance/cost becomes an issue.

## Chunk Size Optimization

### Token Limits

Most embedding models have maximum input lengths:
- BERT-based: 512 tokens (~380 words)
- MPNet: 512 tokens
- Some newer models: 1024+ tokens

**Important**: Text longer than the limit is truncated, losing information.

### Handling Long Chunks

**Strategy 1: Truncation** (Simple)
```python
text[:512]  # Take first 512 tokens
```
- Pros: Simple
- Cons: Loses information from end of text

**Strategy 2: Windowing** (Better)
```python
# Create overlapping windows
windows = create_windows(text, window_size=512, overlap=100)
embeddings = [embed(window) for window in windows]
final_embedding = np.mean(embeddings, axis=0)  # Average
```
- Pros: Captures full content
- Cons: More computations

**Strategy 3: Summarization then Embedding** (Advanced)
```python
summary = summarize(text, max_length=400)
embedding = embed(summary)
```
- Pros: Intelligent compression
- Cons: Requires additional model, may lose details

### Optimal Chunk Sizes by Level

Based on embedding model limits:

| Level | Recommended Size | Rationale |
|-------|------------------|-----------|
| Sentence | 20-100 tokens | Fits well within limits |
| Paragraph | 200-500 tokens | Optimal for most models |
| Section | 800-1500 tokens | Requires windowing or larger models |
| Chapter | 2000-4000 tokens | Requires windowing/summarization |

## Embedding Generation

### Basic Implementation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

# Single text
embedding = model.encode("Machine learning is a subset of AI")

# Batch processing (much faster)
texts = ["text 1", "text 2", "text 3", ...]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

### GPU Acceleration

```python
# Automatic GPU usage
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

# Or specify device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-mpnet-base-v2', device=device)
```

**Performance Gains**: 10-50× faster on GPU vs CPU depending on model size.

### Batch Processing for Large Collections

```python
def embed_large_collection(texts, model, batch_size=32):
    """Efficiently embed large document collections."""

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)
```

### Memory Optimization

For very large collections:

```python
def embed_with_memory_management(texts, model, batch_size=32):
    """Generate embeddings without loading all into memory."""

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, batch_size=batch_size)

        # Store immediately
        store_embeddings(embeddings, ids=range(i, i+len(batch)))

        # Clear from memory
        del embeddings
        torch.cuda.empty_cache()
```

## Distance Metrics

### Cosine Similarity (Recommended)
```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(embedding1, embedding2)
```
- Range: -1 to 1 (typically 0 to 1 for well-trained models)
- Interpretation: Higher = more similar
- Use case: Text similarity, semantic search

### Euclidean Distance (L2)
```python
distance = np.linalg.norm(embedding1 - embedding2)
```
- Range: 0 to ∞
- Interpretation: Lower = more similar
- Use case: When absolute magnitude matters

### Dot Product (Inner Product)
```python
similarity = np.dot(embedding1, embedding2)
```
- Range: -∞ to ∞
- Interpretation: Higher = more similar
- Use case: Fast computation, pre-normalized embeddings

**Recommendation**: Use cosine similarity for text; it's the most interpretable and widely supported.

## Query Embedding Strategies

### Basic Query Embedding
```python
query = "What is machine learning?"
query_embedding = model.encode(query)
```

### Query Expansion
Add context to improve retrieval:

```python
# Add query type context
expanded_query = f"Question: {query} Context: technical definition"
query_embedding = model.encode(expanded_query)
```

### Multi-Query Embedding
For complex queries, generate multiple embeddings:

```python
query = "Compare machine learning and deep learning"

sub_queries = [
    "What is machine learning?",
    "What is deep learning?",
    "Differences between machine learning and deep learning"
]

query_embeddings = model.encode(sub_queries)
# Average or retrieve separately and merge results
```

## Embedding Quality Assessment

### Measuring Retrieval Quality

**Metrics:**
- **Recall@K**: % of relevant docs in top K results
- **MRR (Mean Reciprocal Rank)**: Average rank of first relevant result
- **NDCG (Normalized Discounted Cumulative Gain)**: Weighted relevance score

### Testing Approach

```python
test_cases = [
    {
        "query": "What is machine learning?",
        "relevant_docs": ["doc_123", "doc_456"],
        "chunk_level": "paragraph"
    },
    # More test cases...
]

def evaluate_embeddings(model, test_cases):
    """Evaluate embedding quality on test set."""

    recalls = []

    for case in test_cases:
        query_emb = model.encode(case["query"])
        results = search_database(query_emb, k=10)

        # Check if relevant docs are in results
        relevant_in_results = len(set(case["relevant_docs"]) & set(results))
        recall = relevant_in_results / len(case["relevant_docs"])
        recalls.append(recall)

    return np.mean(recalls)
```

## Embedding Updates and Versioning

### When to Re-Embed

1. **Model Upgrade**: New embedding model with better performance
2. **Content Changes**: Documents modified significantly
3. **Quality Issues**: Current embeddings produce poor results

### Version Management

```python
# Store embedding version with metadata
{
    "chunk_id": "12345",
    "embedding_model": "all-mpnet-base-v2",
    "embedding_version": "v2.1.0",
    "embedding_date": "2024-01-15",
    "embedding_dimensions": 768
}

# When upgrading
def migrate_embeddings(old_collection, new_model):
    """Migrate to new embedding model."""

    texts = old_collection.get()["documents"]
    new_embeddings = new_model.encode(texts, batch_size=32)

    new_collection.add(
        documents=texts,
        embeddings=new_embeddings,
        metadatas=old_collection.get()["metadatas"]
    )
```

## Performance Benchmarks

### Typical Processing Rates

**On consumer GPU (RTX 3090):**
- all-MiniLM-L6-v2: ~14,000 sentences/sec
- all-MiniLM-L12-v2: ~7,000 sentences/sec
- all-mpnet-base-v2: ~2,500 sentences/sec

**On CPU (modern multi-core):**
- all-MiniLM-L6-v2: ~500 sentences/sec
- all-MiniLM-L12-v2: ~250 sentences/sec
- all-mpnet-base-v2: ~100 sentences/sec

**For large collections:**
- 10,000 documents (avg 500 tokens each)
- GPU: 30 minutes - 2 hours depending on model
- CPU: 5-20 hours depending on model

## Advanced Techniques

### Fine-Tuning on Domain Data

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data
train_examples = [
    InputExample(texts=['query', 'positive_doc'], label=1.0),
    InputExample(texts=['query', 'negative_doc'], label=0.0),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Fine-tune
model = SentenceTransformer('all-mpnet-base-v2')
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)
```

### Hybrid Search (Embeddings + Keywords)

Combine semantic search with keyword matching:

```python
def hybrid_search(query, alpha=0.7):
    """Combine semantic and keyword search."""

    # Semantic search
    semantic_results = vector_search(query)

    # Keyword search (BM25, Elasticsearch, etc.)
    keyword_results = keyword_search(query)

    # Combine scores
    final_results = {}
    for doc_id, score in semantic_results.items():
        final_results[doc_id] = alpha * score

    for doc_id, score in keyword_results.items():
        final_results[doc_id] = final_results.get(doc_id, 0) + (1 - alpha) * score

    return sorted(final_results.items(), key=lambda x: x[1], reverse=True)
```

## Related Topics

- **Chunking Strategies**: See hierarchical-document-chunking-strategies.md
- **ChromaDB Storage**: See chromadb-hierarchical-storage.md
- **Query Classification**: See query-classification-routing.md
- **NLP Tools**: See nlp-pipeline-tools-overview.md
