# ChromaDB Hierarchical Storage Strategies

## Overview

ChromaDB is a vector database designed for storing and retrieving embeddings. When implementing hierarchical document chunking, ChromaDB's metadata capabilities and collection structure enable sophisticated multi-level storage and retrieval strategies.

## Core ChromaDB Concepts

### Collections
- Containers for embeddings and their associated metadata
- Each collection can have its own embedding function
- Support for filtering and querying based on metadata

### Documents
- Text chunks stored alongside their embeddings
- Automatically embedded when added to a collection
- Retrieved during similarity searches

### Metadata
- Key-value pairs associated with each document
- Enables filtering before or during vector search
- Critical for implementing hierarchical relationships

## Storage Strategies for Hierarchical Chunks

### Strategy 1: Single Collection with Level Metadata

Store all chunk levels in one collection, differentiated by metadata:

```python
collection = client.create_collection(
    name="hierarchical_documents",
    metadata={"description": "Multi-level document chunks"}
)

collection.add(
    documents=["Chapter 1 full text..."],
    metadatas=[{
        "doc_id": "book_123",
        "chunk_level": "chapter",
        "hierarchy_path": "Chapter 1",
        "token_count": 3500,
        "parent_chunk_id": None,
        "child_chunk_ids": ["section_1_1", "section_1_2"]
    }],
    ids=["chapter_1"]
)
```

**Advantages:**
- Simpler to manage
- Easy cross-level queries
- Single embedding model

**Disadvantages:**
- All chunks use same embedding dimensions
- May not be optimal for different chunk sizes
- Larger collection size

### Strategy 2: Separate Collections Per Level

Create distinct collections for each hierarchical level:

```python
chapter_collection = client.create_collection("chapters")
section_collection = client.create_collection("sections")
paragraph_collection = client.create_collection("paragraphs")
sentence_collection = client.create_collection("sentences")

# Each can use different embedding models optimized for chunk size
```

**Advantages:**
- Optimized embedding models per level
- Faster queries at specific levels
- Better performance tuning

**Disadvantages:**
- More complex management
- Cross-level queries require multiple searches
- Relationship tracking across collections

### Strategy 3: Hybrid Approach (Recommended)

Combine strategies based on use case:
- **Primary collection**: Paragraph-level chunks (most common queries)
- **Supplementary collections**: Chapter/Section (conceptual queries) and Sentence (factual queries)

## Metadata Schema Design

### Essential Metadata Fields

```python
{
    # Identity
    "doc_id": "unique_document_identifier",
    "chunk_id": "unique_chunk_identifier",

    # Hierarchy
    "chunk_level": "document|chapter|section|paragraph|sentence",
    "hierarchy_path": "Chapter 1/Section 1.1/Paragraph 3",
    "parent_chunk_id": "parent_chunk_reference",
    "child_chunk_ids": ["child1", "child2", "child3"],

    # Content Metadata
    "token_count": 450,
    "doc_type": "pdf|html|org-mode|markdown|epub",
    "source_file": "/path/to/original/document.pdf",
    "page_numbers": [15, 16],

    # Processing Metadata
    "coreference_resolved": True,
    "confidence_score": 0.92,
    "processing_date": "2024-01-15T10:30:00Z",
    "needs_review": False
}
```

### Org-roam Zettelkasten Metadata

For org-roam documents, add specialized fields:

```python
{
    # Standard fields...

    # Zettelkasten-specific
    "concept_title": "Machine Learning Applications",
    "relationship_type": "requirement|outcome|definition|example",
    "org_roam_links": ["concept_2.org", "concept_5.org"],
    "backlinks": ["concept_7.org", "concept_12.org"],
    "bibliography_refs": ["<23>", "<45>"],
    "todo_states": ["RESEARCH", "IN_PROGRESS"],
    "tags": ["AI", "healthcare", "research"]
}
```

## Query Patterns

### Basic Similarity Search with Level Filter

```python
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=10,
    where={"chunk_level": "paragraph"}
)
```

### Multi-Level Search

```python
# First search at section level for broad context
section_results = collection.query(
    query_texts=[user_query],
    n_results=5,
    where={"chunk_level": "section"}
)

# Then drill down to paragraphs within relevant sections
section_ids = [r["parent_chunk_id"] for r in section_results]
paragraph_results = collection.query(
    query_texts=[user_query],
    n_results=20,
    where={
        "chunk_level": "paragraph",
        "parent_chunk_id": {"$in": section_ids}
    }
)
```

### Hierarchical Context Reconstruction

```python
def get_chunk_with_context(chunk_id):
    """Retrieve chunk with its full hierarchical context."""

    # Get the chunk
    chunk = collection.get(ids=[chunk_id])
    metadata = chunk["metadatas"][0]

    # Get parent chunks
    parents = []
    current_parent = metadata.get("parent_chunk_id")
    while current_parent:
        parent = collection.get(ids=[current_parent])
        parents.append(parent)
        current_parent = parent["metadatas"][0].get("parent_chunk_id")

    # Get child chunks
    child_ids = metadata.get("child_chunk_ids", [])
    children = collection.get(ids=child_ids) if child_ids else None

    return {
        "chunk": chunk,
        "parents": parents[::-1],  # Root to immediate parent
        "children": children,
        "full_path": metadata.get("hierarchy_path")
    }
```

### Query Routing by Type

```python
def route_query(query_text, query_type):
    """Route queries to appropriate chunk levels."""

    if query_type == "factual":
        # Search sentences and paragraphs
        return collection.query(
            query_texts=[query_text],
            n_results=10,
            where={"chunk_level": {"$in": ["sentence", "paragraph"]}}
        )

    elif query_type == "conceptual":
        # Search sections and chapters
        return collection.query(
            query_texts=[query_text],
            n_results=10,
            where={"chunk_level": {"$in": ["section", "chapter"]}}
        )

    elif query_type == "hybrid":
        # Search all levels with different weights
        all_results = collection.query(
            query_texts=[query_text],
            n_results=30
        )
        # Apply custom ranking based on level and relevance
        return rerank_by_level_and_score(all_results)
```

## Performance Optimization

### Indexing Strategy

ChromaDB uses HNSW (Hierarchical Navigable Small World) indexing by default:

```python
collection = client.create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine",  # or "l2" or "ip"
        "hnsw:construction_ef": 200,
        "hnsw:M": 16
    }
)
```

**Parameters:**
- **hnsw:space**: Distance metric (cosine recommended for text)
- **hnsw:construction_ef**: Higher = better recall, slower indexing
- **hnsw:M**: Higher = better recall, more memory

### Batch Operations

```python
# Efficient batch insertion
collection.add(
    documents=document_list,  # List of 1000s of documents
    metadatas=metadata_list,
    ids=id_list
)

# Efficient batch querying
results = collection.query(
    query_texts=query_list,  # Multiple queries at once
    n_results=10
)
```

### Filtering Performance

```python
# Fast: Index on frequently filtered fields
collection.add(
    metadatas=[{
        "doc_id": "123",  # Frequently filtered
        "chunk_level": "paragraph",  # Frequently filtered
        "rare_field": "value"  # Rarely filtered
    }]
)

# Metadata filtering happens before vector search
# Keep metadata small for better performance
```

## Storage Considerations

### Disk Space

Each document chunk stores:
- Embedding vector (typically 384-1536 dimensions × 4 bytes)
- Original text
- Metadata (JSON)

**Example calculation:**
- 10,000 documents
- 768-dimensional embeddings (3KB each)
- Average text: 500 chars (500 bytes)
- Metadata: ~500 bytes
- **Total**: ~40MB for embeddings + 10MB for text/metadata = 50MB

For hierarchical storage with 4 levels:
- **Storage multiplier**: ~4× (varies based on overlap)
- **Total for 10,000 source documents**: ~200MB

### Backup and Versioning

```python
# Persist collection to disk
client.persist()

# Collections are stored in ChromaDB's data directory
# Recommended: Regular backups of the data directory

# Version control approach
collection_v1 = client.get_or_create_collection("docs_v1")
collection_v2 = client.get_or_create_collection("docs_v2")
```

## Error Handling and Quality Control

### Confidence-Based Storage

```python
def add_chunk_with_confidence(chunk, metadata, confidence):
    """Store chunks with quality metadata."""

    metadata["confidence_score"] = confidence
    metadata["needs_review"] = confidence < 0.75

    if confidence >= 0.85:
        collection.add(documents=[chunk], metadatas=[metadata])
    elif confidence >= 0.65:
        # Add to review queue collection
        review_collection.add(documents=[chunk], metadatas=[metadata])
    else:
        # Flag for manual processing
        manual_queue.add(documents=[chunk], metadatas=[metadata])
```

### Monitoring Queries

```python
# Track query performance
query_log = {
    "query": user_query,
    "timestamp": datetime.now(),
    "results_count": len(results),
    "top_score": results["distances"][0][0],
    "chunk_levels_returned": [r["chunk_level"] for r in results]
}
```

## Integration with Processing Pipeline

### Complete Workflow

```python
def process_and_store_document(doc_path):
    """End-to-end document processing and storage."""

    # 1. Parse document
    doc = parse_document(doc_path)

    # 2. Extract structure
    hierarchy = extract_hierarchy(doc)

    # 3. Apply coreference resolution
    resolved = resolve_coreferences(doc)

    # 4. Create hierarchical chunks
    chunks = create_hierarchical_chunks(resolved, hierarchy)

    # 5. Store in ChromaDB with metadata
    for chunk in chunks:
        collection.add(
            documents=[chunk.text],
            metadatas=[chunk.metadata],
            ids=[chunk.id]
        )

    return len(chunks)
```

## Related Topics

- **Chunking Strategies**: See hierarchical-document-chunking-strategies.md
- **Embedding Selection**: See semantic-search-embedding-strategies.md
- **Query Classification**: See query-classification-routing.md
- **Org-roam Integration**: See org-roam-zettelkasten-semantic-search.md
