# Vector Store Integration for NLP Analysis

## Overview

This document explains how to integrate vector databases (ChromaDB, FAISS) with NLP pipelines for storing and querying document embeddings at multiple hierarchical levels.

## Why Vector Stores?

### Traditional Storage Limitations

**Problem**: SQL databases store entity text but can't efficiently answer:
- "Find documents similar to this one"
- "Which essays discuss related topics?"
- "Group documents by conceptual similarity"

**Solution**: Vector databases store embeddings with fast similarity search.

### Use Cases

1. **Semantic Search**: Find documents by meaning, not keywords
2. **Clustering**: Group similar documents automatically
3. **Deduplication**: Identify near-duplicate content
4. **Recommendation**: Suggest related documents
5. **Multi-Level Analysis**: Compare documents, sections, or paragraphs

## Vector Store Comparison

### ChromaDB (Recommended for Most Use Cases)

**Strengths**:
- Easy to use, Python-first API
- Built-in persistence (no separate server needed)
- Metadata filtering combined with similarity search
- Good for medium-scale datasets (10K-1M documents)
- Active development and documentation

**Limitations**:
- Slower than FAISS for very large datasets
- Less configuration options than specialized systems

### FAISS (Facebook AI Similarity Search)

**Strengths**:
- Extremely fast similarity search
- Optimized for large-scale (millions of vectors)
- Multiple index types for different trade-offs
- GPU acceleration support
- Battle-tested at Facebook/Meta scale

**Limitations**:
- No built-in metadata storage (need external database)
- More complex setup and configuration
- Requires manual index management

### Comparison Matrix

| Feature | ChromaDB | FAISS |
|---------|----------|-------|
| Ease of Use | ★★★★★ | ★★☆☆☆ |
| Speed (small) | ★★★★☆ | ★★★★★ |
| Speed (large) | ★★★☆☆ | ★★★★★ |
| Metadata | ★★★★★ Built-in | ★☆☆☆☆ External |
| Persistence | ★★★★★ Automatic | ★★★☆☆ Manual |
| Scalability | ★★★☆☆ 100K-1M | ★★★★★ 1M-1B+ |
| GPU Support | ☆☆☆☆☆ | ★★★★★ |

**Recommendation**: Start with ChromaDB, migrate to FAISS only if you hit performance limits.

## ChromaDB Implementation

### Installation

```bash
pip install chromadb sentence-transformers
```

### Basic Setup

```python
import chromadb
from chromadb.config import Settings

# Initialize client with persistence
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    chroma_db_impl="duckdb+parquet",  # Fast, persistent storage
    anonymized_telemetry=False
))

# Create collection for documents
collection = client.get_or_create_collection(
    name="student_essays",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)
```

### Storing Document Embeddings

```python
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def store_document(filepath: str, collection):
    """Store document with embedding and metadata."""
    # Read content
    content = Path(filepath).read_text(encoding='utf-8')

    # Generate embedding
    embedding = model.encode(content).tolist()

    # Prepare metadata
    metadata = {
        'filepath': str(filepath),
        'filename': Path(filepath).name,
        'word_count': len(content.split()),
        'char_count': len(content)
    }

    # Store in ChromaDB
    doc_id = Path(filepath).stem  # Use filename as ID
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[metadata]
    )

    print(f"✓ Stored {metadata['filename']} ({metadata['word_count']} words)")

# Example usage
store_document("student_essay_1.md", collection)
store_document("student_essay_2.md", collection)
```

### Querying Similar Documents

```python
def find_similar_documents(query_text: str, collection, n_results: int = 5):
    """Find documents similar to query text."""
    # Embed query
    query_embedding = model.encode(query_text).tolist()

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    similar_docs = []
    for i in range(len(results['ids'][0])):
        similar_docs.append({
            'id': results['ids'][0][i],
            'filename': results['metadatas'][0][i]['filename'],
            'distance': results['distances'][0][i],
            'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
            'content_preview': results['documents'][0][i][:200] + '...'
        })

    return similar_docs

# Example usage
query = "climate change and environmental policy"
similar = find_similar_documents(query, collection, n_results=3)

for doc in similar:
    print(f"{doc['filename']}: {doc['similarity']:.2%} similar")
    print(f"  Preview: {doc['content_preview']}\n")
```

### Metadata Filtering

```python
def search_with_filters(query_text: str, collection, min_words: int = 500):
    """Search with metadata constraints."""
    query_embedding = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"word_count": {"$gte": min_words}},  # Only long documents
        include=["documents", "metadatas", "distances"]
    )

    return results

# Example: Find similar long-form essays
results = search_with_filters(
    "argument about renewable energy",
    collection,
    min_words=1000
)
```

### Updating Embeddings

```python
def update_document(doc_id: str, new_content: str, collection):
    """Update existing document embedding."""
    # Generate new embedding
    new_embedding = model.encode(new_content).tolist()

    # Update metadata
    new_metadata = {
        'word_count': len(new_content.split()),
        'char_count': len(new_content),
        'updated_at': datetime.utcnow().isoformat()
    }

    # ChromaDB doesn't have update, so delete and re-add
    collection.delete(ids=[doc_id])
    collection.add(
        ids=[doc_id],
        embeddings=[new_embedding],
        documents=[new_content],
        metadatas=[new_metadata]
    )

# Example usage
update_document("essay_1", "Revised content...", collection)
```

## Hierarchical Embeddings in ChromaDB

### Storing Multiple Levels

Store document-level, section-level, and paragraph-level embeddings in same database:

```python
def store_hierarchical_document(filepath: str, client):
    """Store document with embeddings at multiple levels."""
    content = Path(filepath).read_text(encoding='utf-8')
    doc_id = Path(filepath).stem

    # Get or create collections for each level
    doc_collection = client.get_or_create_collection("documents")
    section_collection = client.get_or_create_collection("sections")
    para_collection = client.get_or_create_collection("paragraphs")

    # 1. Document-level embedding
    doc_embedding = model.encode(content).tolist()
    doc_collection.add(
        ids=[doc_id],
        embeddings=[doc_embedding],
        documents=[content],
        metadatas=[{'filepath': str(filepath), 'level': 'document'}]
    )

    # 2. Section-level embeddings
    sections = split_into_sections(content)  # Markdown heading-based
    for i, section in enumerate(sections):
        section_id = f"{doc_id}_sec_{i}"
        section_embedding = model.encode(section['content']).tolist()

        section_collection.add(
            ids=[section_id],
            embeddings=[section_embedding],
            documents=[section['content']],
            metadatas=[{
                'document_id': doc_id,
                'section_index': i,
                'heading': section['heading'],
                'level': 'section'
            }]
        )

    # 3. Paragraph-level embeddings
    paragraphs = content.split('\n\n')
    for i, para in enumerate(paragraphs):
        if len(para.strip()) < 50:  # Skip very short paragraphs
            continue

        para_id = f"{doc_id}_para_{i}"
        para_embedding = model.encode(para).tolist()

        para_collection.add(
            ids=[para_id],
            embeddings=[para_embedding],
            documents=[para],
            metadatas=[{
                'document_id': doc_id,
                'paragraph_index': i,
                'level': 'paragraph'
            }]
        )

    print(f"✓ Stored {doc_id}: {len(sections)} sections, {len(paragraphs)} paragraphs")

def split_into_sections(content: str) -> list:
    """Split markdown by headings."""
    sections = []
    current_section = {'heading': None, 'content': []}

    for line in content.split('\n'):
        if line.startswith('#'):
            if current_section['content']:
                sections.append({
                    'heading': current_section['heading'],
                    'content': '\n'.join(current_section['content'])
                })
            current_section = {'heading': line.strip('# '), 'content': []}
        else:
            current_section['content'].append(line)

    # Add final section
    if current_section['content']:
        sections.append({
            'heading': current_section['heading'],
            'content': '\n'.join(current_section['content'])
        })

    return sections

# Example usage
store_hierarchical_document("student_essay_1.md", client)
```

### Multi-Level Querying

```python
def search_at_level(query_text: str, level: str, client, n_results: int = 5):
    """Search specific hierarchical level."""
    collection_name = {
        'document': 'documents',
        'section': 'sections',
        'paragraph': 'paragraphs'
    }[level]

    collection = client.get_collection(collection_name)
    query_embedding = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    return results

# Example: Search at different levels
query = "renewable energy policy arguments"

# Document-level: Find relevant essays
docs = search_at_level(query, 'document', client, n_results=3)
print("Most relevant documents:")
for i, doc_id in enumerate(docs['ids'][0]):
    print(f"  {i+1}. {docs['metadatas'][0][i]['filepath']}")

# Section-level: Find specific sections
sections = search_at_level(query, 'section', client, n_results=5)
print("\nMost relevant sections:")
for i, sec_id in enumerate(sections['ids'][0]):
    meta = sections['metadatas'][0][i]
    print(f"  {i+1}. {meta['document_id']} - {meta['heading']}")

# Paragraph-level: Find specific passages
paragraphs = search_at_level(query, 'paragraph', client, n_results=5)
print("\nMost relevant paragraphs:")
for i, para in enumerate(paragraphs['documents'][0]):
    print(f"  {i+1}. {para[:100]}...")
```

### Cross-Level Analysis

```python
def find_document_from_paragraph(para_id: str, client):
    """Trace paragraph back to parent document."""
    para_collection = client.get_collection("paragraphs")
    para_result = para_collection.get(ids=[para_id], include=["metadatas"])

    doc_id = para_result['metadatas'][0]['document_id']

    doc_collection = client.get_collection("documents")
    doc_result = doc_collection.get(ids=[doc_id], include=["documents", "metadatas"])

    return {
        'document_id': doc_id,
        'filepath': doc_result['metadatas'][0]['filepath'],
        'full_content': doc_result['documents'][0]
    }

# Example: Find context for relevant paragraph
paragraphs = search_at_level("carbon emissions reduction", 'paragraph', client, n_results=1)
top_para_id = paragraphs['ids'][0][0]

context = find_document_from_paragraph(top_para_id, client)
print(f"This paragraph is from: {context['filepath']}")
```

## FAISS Implementation

### Installation

```bash
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install sentence-transformers numpy
```

### Basic Setup

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384  # Dimension of all-MiniLM-L6-v2

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)

# For cosine similarity, normalize vectors:
# index = faiss.IndexFlatIP(embedding_dim)  # Inner product = cosine if normalized

# Metadata storage (FAISS doesn't store this)
metadata_store = {}
```

### Storing Embeddings

```python
def add_document_to_faiss(filepath: str, index, metadata_store, next_id: int):
    """Add document to FAISS index."""
    content = Path(filepath).read_text(encoding='utf-8')

    # Generate embedding
    embedding = model.encode(content)

    # Normalize for cosine similarity
    faiss.normalize_L2(embedding.reshape(1, -1))

    # Add to index
    index.add(embedding.reshape(1, -1).astype('float32'))

    # Store metadata separately
    metadata_store[next_id] = {
        'filepath': str(filepath),
        'filename': Path(filepath).name,
        'content': content,
        'word_count': len(content.split())
    }

    return next_id + 1

# Example: Add multiple documents
next_id = 0
for essay_file in Path("essays").glob("*.md"):
    next_id = add_document_to_faiss(essay_file, index, metadata_store, next_id)

print(f"Added {next_id} documents to FAISS index")
```

### Querying

```python
def search_faiss(query_text: str, index, metadata_store, k: int = 5):
    """Search FAISS index for similar documents."""
    # Embed query
    query_embedding = model.encode(query_text)

    # Normalize
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    # Search
    distances, indices = index.search(
        query_embedding.reshape(1, -1).astype('float32'),
        k
    )

    # Retrieve metadata
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:  # No more results
            break

        similarity = 1 - distances[0][i]  # Convert distance to similarity
        results.append({
            'index': int(idx),
            'similarity': float(similarity),
            'filename': metadata_store[idx]['filename'],
            'filepath': metadata_store[idx]['filepath'],
            'content_preview': metadata_store[idx]['content'][:200] + '...'
        })

    return results

# Example usage
query = "climate change mitigation strategies"
results = search_faiss(query, index, metadata_store, k=3)

for result in results:
    print(f"{result['filename']}: {result['similarity']:.2%} similar")
```

### Persistence

```python
def save_faiss_index(index, metadata_store, index_path: str, metadata_path: str):
    """Save FAISS index and metadata to disk."""
    # Save index
    faiss.write_index(index, index_path)

    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_store, f)

    print(f"✓ Saved index to {index_path}")

def load_faiss_index(index_path: str, metadata_path: str):
    """Load FAISS index and metadata from disk."""
    # Load index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata_store = pickle.load(f)

    print(f"✓ Loaded index with {index.ntotal} vectors")

    return index, metadata_store

# Example usage
save_faiss_index(index, metadata_store, "essays.index", "essays_metadata.pkl")

# Later session
index, metadata_store = load_faiss_index("essays.index", "essays_metadata.pkl")
```

### Advanced: IVF Index for Large Datasets

For millions of vectors, use inverted file index:

```python
def create_ivf_index(embedding_dim: int, n_clusters: int = 100):
    """Create IVF (Inverted File) index for fast approximate search."""
    # Quantizer (coarse clustering)
    quantizer = faiss.IndexFlatL2(embedding_dim)

    # IVF index
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)

    return index

def train_ivf_index(index, training_embeddings):
    """Train IVF index on sample data."""
    # IVF requires training on representative data
    training_embeddings = np.array(training_embeddings, dtype='float32')
    faiss.normalize_L2(training_embeddings)

    index.train(training_embeddings)
    print(f"✓ Trained IVF index on {len(training_embeddings)} vectors")

# Example
ivf_index = create_ivf_index(embedding_dim=384, n_clusters=100)

# Generate training data (sample of documents)
training_docs = list(Path("essays").glob("*.md"))[:500]
training_embeddings = [model.encode(Path(f).read_text()) for f in training_docs]
train_ivf_index(ivf_index, training_embeddings)

# Set search parameters
ivf_index.nprobe = 10  # Search 10 clusters (speed/accuracy trade-off)

# Now add all documents
for essay_file in Path("essays").glob("*.md"):
    next_id = add_document_to_faiss(essay_file, ivf_index, metadata_store, next_id)
```

## Hybrid Approach: FAISS + SQLite

Best of both worlds - FAISS for vector search, SQLite for metadata:

```python
import sqlite3
import faiss
import numpy as np

class HybridVectorStore:
    """FAISS for vectors, SQLite for metadata."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Cosine similarity
        self.db = sqlite3.connect("vector_metadata.db")
        self._create_tables()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _create_tables(self):
        """Initialize database schema."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                vector_id INTEGER PRIMARY KEY,
                filepath TEXT NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                word_count INTEGER,
                added_at TEXT NOT NULL
            )
        """)
        self.db.commit()

    def add_document(self, filepath: str):
        """Add document with metadata."""
        content = Path(filepath).read_text(encoding='utf-8')

        # Generate and normalize embedding
        embedding = self.model.encode(content)
        faiss.normalize_L2(embedding.reshape(1, -1))

        # Add to FAISS
        vector_id = self.index.ntotal
        self.index.add(embedding.reshape(1, -1).astype('float32'))

        # Store metadata in SQLite
        self.db.execute("""
            INSERT INTO documents (vector_id, filepath, filename, content, word_count, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            vector_id,
            str(filepath),
            Path(filepath).name,
            content,
            len(content.split()),
            datetime.utcnow().isoformat()
        ))
        self.db.commit()

        return vector_id

    def search(self, query_text: str, k: int = 5, min_words: int = 0):
        """Search with metadata filtering."""
        # Vector search
        query_embedding = self.model.encode(query_text)
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k * 2  # Get more results for filtering
        )

        # Fetch metadata and filter
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                break

            cursor = self.db.execute("""
                SELECT filepath, filename, content, word_count
                FROM documents
                WHERE vector_id = ? AND word_count >= ?
            """, (int(idx), min_words))

            row = cursor.fetchone()
            if row:
                similarity = float(distances[0][i])
                results.append({
                    'vector_id': int(idx),
                    'similarity': similarity,
                    'filepath': row[0],
                    'filename': row[1],
                    'content_preview': row[2][:200] + '...',
                    'word_count': row[3]
                })

            if len(results) >= k:
                break

        return results

    def save(self, index_path: str):
        """Persist FAISS index (SQLite auto-persists)."""
        faiss.write_index(self.index, index_path)

    def load(self, index_path: str):
        """Load FAISS index (SQLite auto-loads)."""
        self.index = faiss.read_index(index_path)

# Example usage
store = HybridVectorStore()

# Add documents
for essay_file in Path("essays").glob("*.md"):
    store.add_document(essay_file)

# Search with filtering
results = store.search(
    "environmental policy arguments",
    k=5,
    min_words=500  # Only essays with 500+ words
)

for result in results:
    print(f"{result['filename']}: {result['similarity']:.2%} "
          f"({result['word_count']} words)")

# Save for later
store.save("essays.faiss")
```

## Performance Comparison

### Benchmark Setup

```python
import time
from pathlib import Path

def benchmark_search(query: str, n_queries: int = 100):
    """Compare ChromaDB vs FAISS search speed."""
    # Setup
    chroma_collection = client.get_collection("student_essays")
    query_embedding = model.encode(query).tolist()

    # Benchmark ChromaDB
    start = time.time()
    for _ in range(n_queries):
        chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
    chroma_time = time.time() - start

    # Benchmark FAISS
    faiss_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(faiss_embedding)

    start = time.time()
    for _ in range(n_queries):
        index.search(faiss_embedding, 5)
    faiss_time = time.time() - start

    print(f"ChromaDB: {chroma_time:.3f}s ({n_queries/chroma_time:.1f} queries/sec)")
    print(f"FAISS:    {faiss_time:.3f}s ({n_queries/faiss_time:.1f} queries/sec)")
    print(f"Speedup:  {chroma_time/faiss_time:.1f}x")

# Example results (typical)
# ChromaDB: 2.450s (40.8 queries/sec)
# FAISS:    0.182s (549.5 queries/sec)
# Speedup:  13.5x
```

### When to Use What

**Use ChromaDB if**:
- Dataset < 100K documents
- Need metadata filtering
- Want simple setup
- Rapid prototyping

**Use FAISS if**:
- Dataset > 100K documents
- Need maximum speed
- Have GPU available
- Building production system

**Use Hybrid (FAISS + SQL) if**:
- Large dataset but need metadata
- Complex filtering requirements
- Want FAISS speed with SQL flexibility

## Best Practices

### 1. Normalize Embeddings for Cosine Similarity

```python
# ✅ Good: Normalized for cosine similarity
faiss.normalize_L2(embedding.reshape(1, -1))

# ❌ Bad: Unnormalized (L2 distance ≠ cosine)
index.add(embedding.reshape(1, -1))
```

### 2. Batch Processing

```python
# ✅ Good: Batch additions
embeddings = np.array([model.encode(doc) for doc in documents])
faiss.normalize_L2(embeddings)
index.add(embeddings)

# ❌ Bad: One at a time (slow)
for doc in documents:
    emb = model.encode(doc)
    index.add(emb.reshape(1, -1))
```

### 3. Persist Regularly

```python
# ✅ Good: Save after significant changes
if documents_added % 100 == 0:
    save_faiss_index(index, metadata_store, "backup.index", "backup.pkl")

# ❌ Bad: Only save at end (risk data loss)
```

### 4. Version Your Embeddings

```python
# ✅ Good: Track embedding model version
metadata = {
    'filepath': filepath,
    'embedding_model': 'all-MiniLM-L6-v2',
    'embedding_version': '1.0'
}

# ❌ Bad: Can't reproduce if model changes
```

## Related Documentation

- [NLP Tools Comparison](nlp-tools-comparison.org)
- [Document Provenance Tracking](nlp-document-provenance-tracking.md)
- [Hierarchical Embeddings](nlp-hierarchical-embeddings.md)
- [Semantic Search Strategies](semantic-search-embedding-strategies.md)
