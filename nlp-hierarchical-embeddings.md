# Hierarchical Document Embeddings

## Overview

This document explains how to generate and use multi-level embeddings (document, section, paragraph, sentence) for comprehensive semantic analysis of student essays and academic texts.

## Why Hierarchical Embeddings?

### Single-Level Limitations

**Problem with document-level only**:
- Query: "arguments about carbon taxes"
- Result: Entire 5000-word essay (but carbon taxes only in §3)
- User must read everything to find relevant part

**Problem with sentence-level only**:
- Loses broader context
- Can't answer "which documents discuss topic X?"
- Too granular for document clustering

### Hierarchical Solution

Multiple embedding levels enable different query types:

1. **Document-level**: "Which essays discuss climate policy?" → Find relevant documents
2. **Section-level**: "What are the main arguments?" → Find key sections
3. **Paragraph-level**: "Find supporting evidence" → Locate specific passages
4. **Sentence-level**: "Extract exact claims" → Pinpoint statements

## Embedding Levels Explained

### 1. Document-Level Embeddings

**Granularity**: Entire document (full essay, paper, article)

**Use Cases**:
- Document clustering (group similar essays)
- Duplicate detection
- Topic classification
- Document recommendation

**Generation**:

```python
from sentence_transformers import SentenceTransformer
from pathlib import Path

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_document_embedding(filepath: str):
    """Generate embedding for entire document."""
    content = Path(filepath).read_text(encoding='utf-8')

    # Single embedding for full content
    embedding = model.encode(content, show_progress_bar=False)

    return {
        'level': 'document',
        'filepath': filepath,
        'embedding': embedding.tolist(),
        'char_count': len(content),
        'word_count': len(content.split())
    }

# Example
doc_emb = generate_document_embedding("student_essay_1.md")
print(f"Document embedding: {len(doc_emb['embedding'])}-dimensional vector")
```

**Limitations**:
- Max input length (~512 tokens for most models)
- For long documents, content gets truncated
- Loses fine-grained details

### 2. Section-Level Embeddings

**Granularity**: Logical document sections (based on markdown headings)

**Use Cases**:
- Find specific sections across documents
- Compare argumentation structure
- Section-level similarity
- Extract relevant discussion topics

**Generation**:

```python
def generate_section_embeddings(filepath: str):
    """Generate embeddings for each markdown section."""
    content = Path(filepath).read_text(encoding='utf-8')
    sections = split_by_headings(content)

    section_embeddings = []
    for i, section in enumerate(sections):
        embedding = model.encode(section['content'], show_progress_bar=False)

        section_embeddings.append({
            'level': 'section',
            'document_id': Path(filepath).stem,
            'section_index': i,
            'heading': section['heading'],
            'heading_level': section['level'],
            'embedding': embedding.tolist(),
            'word_count': len(section['content'].split())
        })

    return section_embeddings

def split_by_headings(content: str):
    """Split markdown content by headings."""
    sections = []
    current_section = None

    for line in content.split('\n'):
        if line.startswith('#'):
            # New section
            if current_section:
                sections.append(current_section)

            heading_level = len(line) - len(line.lstrip('#'))
            heading_text = line.lstrip('# ').strip()

            current_section = {
                'heading': heading_text,
                'level': heading_level,
                'content': []
            }
        elif current_section is not None:
            current_section['content'].append(line)

    # Add final section
    if current_section:
        current_section['content'] = '\n'.join(current_section['content'])
        sections.append(current_section)

    return sections

# Example
section_embs = generate_section_embeddings("student_essay_1.md")
print(f"Generated {len(section_embs)} section embeddings")
for sec in section_embs:
    print(f"  §{sec['section_index']}: {sec['heading']} ({sec['word_count']} words)")
```

### 3. Paragraph-Level Embeddings

**Granularity**: Individual paragraphs

**Use Cases**:
- Find specific arguments or evidence
- Paragraph-level similarity
- Extract quotes and passages
- Detailed content analysis

**Generation**:

```python
def generate_paragraph_embeddings(filepath: str):
    """Generate embeddings for each paragraph."""
    content = Path(filepath).read_text(encoding='utf-8')
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]

    paragraph_embeddings = []
    for i, paragraph in enumerate(paragraphs):
        embedding = model.encode(paragraph, show_progress_bar=False)

        paragraph_embeddings.append({
            'level': 'paragraph',
            'document_id': Path(filepath).stem,
            'paragraph_index': i,
            'embedding': embedding.tolist(),
            'text': paragraph,
            'char_count': len(paragraph),
            'sentence_count': len([s for s in paragraph.split('.') if s.strip()])
        })

    return paragraph_embeddings

# Example
para_embs = generate_paragraph_embeddings("student_essay_1.md")
print(f"Generated {len(para_embs)} paragraph embeddings")

# Find most similar paragraphs across documents
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_paragraphs(query: str, all_paragraph_embeddings: list, top_k: int = 5):
    """Find most similar paragraphs to query."""
    query_embedding = model.encode(query)

    # Calculate similarities
    similarities = []
    for para_emb in all_paragraph_embeddings:
        emb = np.array(para_emb['embedding'])
        sim = cosine_similarity([query_embedding], [emb])[0][0]
        similarities.append({
            'paragraph': para_emb,
            'similarity': sim
        })

    # Sort and return top-k
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

# Example query
results = find_similar_paragraphs("carbon tax effectiveness", para_embs, top_k=3)
for i, result in enumerate(results):
    print(f"{i+1}. Similarity: {result['similarity']:.2%}")
    print(f"   {result['paragraph']['text'][:100]}...\n")
```

### 4. Sentence-Level Embeddings

**Granularity**: Individual sentences

**Use Cases**:
- Exact claim extraction
- Fine-grained similarity
- Sentence alignment across documents
- Quote attribution

**Generation**:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def generate_sentence_embeddings(filepath: str):
    """Generate embeddings for each sentence."""
    content = Path(filepath).read_text(encoding='utf-8')
    doc = nlp(content)

    sentence_embeddings = []
    for i, sent in enumerate(doc.sents):
        if len(sent.text.strip()) < 10:  # Skip very short sentences
            continue

        embedding = model.encode(sent.text, show_progress_bar=False)

        sentence_embeddings.append({
            'level': 'sentence',
            'document_id': Path(filepath).stem,
            'sentence_index': i,
            'embedding': embedding.tolist(),
            'text': sent.text,
            'start_char': sent.start_char,
            'end_char': sent.end_char
        })

    return sentence_embeddings

# Example
sent_embs = generate_sentence_embeddings("student_essay_1.md")
print(f"Generated {len(sent_embs)} sentence embeddings")

# Find exact matching sentences
def find_matching_sentences(sentence: str, all_sentence_embeddings: list, threshold: float = 0.85):
    """Find near-duplicate or highly similar sentences."""
    query_embedding = model.encode(sentence)

    matches = []
    for sent_emb in all_sentence_embeddings:
        emb = np.array(sent_emb['embedding'])
        sim = cosine_similarity([query_embedding], [emb])[0][0]

        if sim >= threshold:
            matches.append({
                'sentence': sent_emb,
                'similarity': sim
            })

    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return matches

# Example: Find where students make similar claims
claim = "Climate change requires immediate action"
matches = find_matching_sentences(claim, sent_embs, threshold=0.80)

print(f"Found {len(matches)} similar sentences:")
for match in matches:
    print(f"  {match['similarity']:.2%}: {match['sentence']['text']}")
```

## Complete Hierarchical System

### Unified Embedding Generator

```python
class HierarchicalEmbedder:
    """Generate embeddings at all levels for a document."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")

    def embed_document(self, filepath: str) -> dict:
        """Generate all hierarchical embeddings for document."""
        content = Path(filepath).read_text(encoding='utf-8')
        doc_id = Path(filepath).stem

        print(f"Embedding {doc_id} at all levels...")

        # Document-level
        doc_embedding = self._embed_document_level(content, doc_id)

        # Section-level
        section_embeddings = self._embed_section_level(content, doc_id)

        # Paragraph-level
        paragraph_embeddings = self._embed_paragraph_level(content, doc_id)

        # Sentence-level
        sentence_embeddings = self._embed_sentence_level(content, doc_id)

        return {
            'document_id': doc_id,
            'filepath': str(filepath),
            'document': doc_embedding,
            'sections': section_embeddings,
            'paragraphs': paragraph_embeddings,
            'sentences': sentence_embeddings,
            'stats': {
                'section_count': len(section_embeddings),
                'paragraph_count': len(paragraph_embeddings),
                'sentence_count': len(sentence_embeddings)
            }
        }

    def _embed_document_level(self, content: str, doc_id: str):
        """Document-level embedding."""
        # Truncate if too long (most models have ~512 token limit)
        truncated = content[:8000]  # ~2000 tokens
        embedding = self.model.encode(truncated, show_progress_bar=False)

        return {
            'level': 'document',
            'document_id': doc_id,
            'embedding': embedding.tolist(),
            'was_truncated': len(content) > 8000
        }

    def _embed_section_level(self, content: str, doc_id: str):
        """Section-level embeddings."""
        sections = split_by_headings(content)
        section_embeddings = []

        for i, section in enumerate(sections):
            embedding = self.model.encode(section['content'], show_progress_bar=False)
            section_embeddings.append({
                'level': 'section',
                'document_id': doc_id,
                'section_index': i,
                'heading': section['heading'],
                'embedding': embedding.tolist()
            })

        return section_embeddings

    def _embed_paragraph_level(self, content: str, doc_id: str):
        """Paragraph-level embeddings."""
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        paragraph_embeddings = []

        for i, paragraph in enumerate(paragraphs):
            embedding = self.model.encode(paragraph, show_progress_bar=False)
            paragraph_embeddings.append({
                'level': 'paragraph',
                'document_id': doc_id,
                'paragraph_index': i,
                'embedding': embedding.tolist(),
                'text': paragraph
            })

        return paragraph_embeddings

    def _embed_sentence_level(self, content: str, doc_id: str):
        """Sentence-level embeddings."""
        doc = self.nlp(content)
        sentence_embeddings = []

        for i, sent in enumerate(doc.sents):
            if len(sent.text.strip()) < 10:
                continue

            embedding = self.model.encode(sent.text, show_progress_bar=False)
            sentence_embeddings.append({
                'level': 'sentence',
                'document_id': doc_id,
                'sentence_index': i,
                'embedding': embedding.tolist(),
                'text': sent.text
            })

        return sentence_embeddings

# Example usage
embedder = HierarchicalEmbedder()
result = embedder.embed_document("student_essay_1.md")

print(f"Generated hierarchical embeddings:")
print(f"  Document: 1 embedding")
print(f"  Sections: {result['stats']['section_count']} embeddings")
print(f"  Paragraphs: {result['stats']['paragraph_count']} embeddings")
print(f"  Sentences: {result['stats']['sentence_count']} embeddings")
```

### Storing Hierarchical Embeddings

```python
import chromadb
from chromadb.config import Settings

class HierarchicalVectorStore:
    """Store and query hierarchical embeddings."""

    def __init__(self, persist_directory: str = "./hierarchical_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet"
        ))

        # Create separate collections for each level
        self.doc_collection = self.client.get_or_create_collection("documents")
        self.section_collection = self.client.get_or_create_collection("sections")
        self.para_collection = self.client.get_or_create_collection("paragraphs")
        self.sent_collection = self.client.get_or_create_collection("sentences")

    def store_hierarchical_document(self, embeddings: dict):
        """Store all levels of embeddings."""
        doc_id = embeddings['document_id']

        # Store document-level
        self.doc_collection.add(
            ids=[doc_id],
            embeddings=[embeddings['document']['embedding']],
            metadatas=[{
                'filepath': embeddings['filepath'],
                'level': 'document'
            }]
        )

        # Store section-level
        for section in embeddings['sections']:
            section_id = f"{doc_id}_sec_{section['section_index']}"
            self.section_collection.add(
                ids=[section_id],
                embeddings=[section['embedding']],
                metadatas=[{
                    'document_id': doc_id,
                    'section_index': section['section_index'],
                    'heading': section['heading'],
                    'level': 'section'
                }]
            )

        # Store paragraph-level
        for paragraph in embeddings['paragraphs']:
            para_id = f"{doc_id}_para_{paragraph['paragraph_index']}"
            self.para_collection.add(
                ids=[para_id],
                embeddings=[paragraph['embedding']],
                documents=[paragraph['text']],
                metadatas=[{
                    'document_id': doc_id,
                    'paragraph_index': paragraph['paragraph_index'],
                    'level': 'paragraph'
                }]
            )

        # Store sentence-level
        for sentence in embeddings['sentences']:
            sent_id = f"{doc_id}_sent_{sentence['sentence_index']}"
            self.sent_collection.add(
                ids=[sent_id],
                embeddings=[sentence['embedding']],
                documents=[sentence['text']],
                metadatas=[{
                    'document_id': doc_id,
                    'sentence_index': sentence['sentence_index'],
                    'level': 'sentence'
                }]
            )

        print(f"✓ Stored {doc_id} at all hierarchical levels")

    def query_at_level(self, query: str, level: str, n_results: int = 5):
        """Query specific hierarchical level."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query).tolist()

        collection_map = {
            'document': self.doc_collection,
            'section': self.section_collection,
            'paragraph': self.para_collection,
            'sentence': self.sent_collection
        }

        collection = collection_map[level]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results

# Example usage
embedder = HierarchicalEmbedder()
store = HierarchicalVectorStore()

# Process and store documents
for essay_file in Path("essays").glob("*.md"):
    embeddings = embedder.embed_document(essay_file)
    store.store_hierarchical_document(embeddings)

# Query at different levels
query = "renewable energy policy arguments"

print("=== Document-Level Results ===")
doc_results = store.query_at_level(query, 'document', n_results=3)
for i, doc_id in enumerate(doc_results['ids'][0]):
    print(f"{i+1}. {doc_id}")

print("\n=== Section-Level Results ===")
sec_results = store.query_at_level(query, 'section', n_results=5)
for i, meta in enumerate(sec_results['metadatas'][0]):
    print(f"{i+1}. {meta['document_id']} - §{meta['section_index']}: {meta['heading']}")

print("\n=== Paragraph-Level Results ===")
para_results = store.query_at_level(query, 'paragraph', n_results=3)
for i, text in enumerate(para_results['documents'][0]):
    print(f"{i+1}. {text[:150]}...\n")
```

## Query Routing Strategy

Automatically route queries to optimal embedding level:

```python
class QueryRouter:
    """Route queries to appropriate embedding level."""

    def __init__(self, store: HierarchicalVectorStore):
        self.store = store
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def classify_query(self, query: str) -> str:
        """Determine optimal embedding level for query."""
        query_lower = query.lower()

        # Document-level indicators
        if any(kw in query_lower for kw in ['documents about', 'essays on', 'papers discussing', 'which documents']):
            return 'document'

        # Section-level indicators
        if any(kw in query_lower for kw in ['section', 'argument', 'discussion', 'main points']):
            return 'section'

        # Sentence-level indicators
        if any(kw in query_lower for kw in ['exact', 'quote', 'claim', 'sentence', 'statement']):
            return 'sentence'

        # Default to paragraph (good balance)
        return 'paragraph'

    def smart_search(self, query: str, n_results: int = 5):
        """Search with automatic level selection."""
        level = self.classify_query(query)
        print(f"Routing to {level} level")

        results = self.store.query_at_level(query, level, n_results)

        return {
            'query': query,
            'level': level,
            'results': results
        }

# Example usage
router = QueryRouter(store)

# Automatically routed to document level
results1 = router.smart_search("Which essays discuss climate change?")

# Automatically routed to section level
results2 = router.smart_search("What are the main arguments about carbon taxes?")

# Automatically routed to sentence level
results3 = router.smart_search("Find exact claims about renewable energy")

# Automatically routed to paragraph level (default)
results4 = router.smart_search("evidence for policy effectiveness")
```

## Cross-Level Analysis

### Finding Context for Fine-Grained Results

```python
def find_context_for_sentence(sentence_id: str, store: HierarchicalVectorStore):
    """Find document and section context for a sentence."""
    # Get sentence metadata
    sent_result = store.sent_collection.get(ids=[sentence_id], include=["metadatas", "documents"])
    sent_meta = sent_result['metadatas'][0]
    sent_text = sent_result['documents'][0]

    doc_id = sent_meta['document_id']

    # Get parent document
    doc_result = store.doc_collection.get(ids=[doc_id], include=["metadatas"])
    doc_meta = doc_result['metadatas'][0]

    # Find which section contains this sentence
    section_results = store.section_collection.get(
        where={"document_id": doc_id},
        include=["metadatas"]
    )

    return {
        'sentence': sent_text,
        'document': {
            'id': doc_id,
            'filepath': doc_meta['filepath']
        },
        'sections': [
            {'index': meta['section_index'], 'heading': meta['heading']}
            for meta in section_results['metadatas']
        ]
    }

# Example: Trace sentence back to source
sentence_id = "essay_1_sent_42"
context = find_context_for_sentence(sentence_id, store)

print(f"Sentence: {context['sentence']}")
print(f"From document: {context['document']['filepath']}")
print(f"Sections in document: {', '.join(s['heading'] for s in context['sections'])}")
```

### Aggregating Fine-Grained Results

```python
def aggregate_sentence_to_document(query: str, store: HierarchicalVectorStore, threshold: float = 0.75):
    """Find documents by aggregating sentence-level matches."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()

    # Get many sentence-level matches
    sent_results = store.sent_collection.query(
        query_embeddings=[query_embedding],
        n_results=50,
        include=["metadatas", "distances"]
    )

    # Group by document
    from collections import Counter
    doc_match_counts = Counter()

    for i, meta in enumerate(sent_results['metadatas'][0]):
        similarity = 1 - sent_results['distances'][0][i]
        if similarity >= threshold:
            doc_match_counts[meta['document_id']] += 1

    # Rank documents by number of matching sentences
    ranked_docs = doc_match_counts.most_common(5)

    print(f"Documents with most relevant sentences:")
    for doc_id, count in ranked_docs:
        print(f"  {doc_id}: {count} matching sentences")

    return ranked_docs

# Example: Find documents with most mentions of topic
aggregate_sentence_to_document("carbon pricing mechanisms", store, threshold=0.80)
```

## Performance Optimization

### Caching Embeddings

```python
import hashlib
import pickle
from pathlib import Path

class CachedEmbedder:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedder = HierarchicalEmbedder()

    def _get_cache_key(self, filepath: str) -> str:
        """Generate cache key from file content hash."""
        content = Path(filepath).read_text(encoding='utf-8')
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{Path(filepath).stem}_{content_hash}.pkl"

    def embed_document(self, filepath: str, use_cache: bool = True):
        """Embed document with caching."""
        cache_key = self._get_cache_key(filepath)
        cache_file = self.cache_dir / cache_key

        # Check cache
        if use_cache and cache_file.exists():
            with open(cache_file, 'rb') as f:
                print(f"✓ Loaded cached embeddings for {Path(filepath).name}")
                return pickle.load(f)

        # Generate embeddings
        embeddings = self.embedder.embed_document(filepath)

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)

        return embeddings

# Example: Second call uses cache
cached_embedder = CachedEmbedder()
embeddings1 = cached_embedder.embed_document("essay.md")  # Generates
embeddings2 = cached_embedder.embed_document("essay.md")  # Uses cache
```

### Batch Processing

```python
def batch_embed_documents(filepaths: list, batch_size: int = 32):
    """Process multiple documents efficiently."""
    embedder = HierarchicalEmbedder()
    all_embeddings = []

    for i in range(0, len(filepaths), batch_size):
        batch = filepaths[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(filepaths)-1)//batch_size + 1}")

        batch_embeddings = [embedder.embed_document(fp) for fp in batch]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

# Example
essay_files = list(Path("essays").glob("*.md"))
all_embeddings = batch_embed_documents(essay_files, batch_size=10)
```

## Use Case Examples

### 1. Finding Similar Arguments Across Documents

```python
def find_argument_agreements(query_argument: str, store: HierarchicalVectorStore):
    """Find where students make similar arguments."""
    # Search at paragraph level (good for arguments)
    results = store.query_at_level(query_argument, 'paragraph', n_results=10)

    agreements = []
    for i, text in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        similarity = 1 - results['distances'][0][i]

        if similarity >= 0.75:  # High similarity threshold
            agreements.append({
                'document_id': meta['document_id'],
                'paragraph_index': meta['paragraph_index'],
                'text': text,
                'similarity': similarity
            })

    return agreements

# Example
query = "Carbon taxes are effective at reducing emissions"
agreements = find_argument_agreements(query, store)

print(f"Found {len(agreements)} similar arguments:")
for arg in agreements:
    print(f"\n{arg['document_id']} (§{arg['paragraph_index']}): {arg['similarity']:.2%}")
    print(f"  {arg['text'][:200]}...")
```

### 2. Document Clustering by Topic

```python
from sklearn.cluster import KMeans
import numpy as np

def cluster_documents_by_topic(store: HierarchicalVectorStore, n_clusters: int = 5):
    """Cluster documents using document-level embeddings."""
    # Get all document embeddings
    all_docs = store.doc_collection.get(include=["embeddings", "metadatas"])

    embeddings_matrix = np.array(all_docs['embeddings'])
    doc_ids = all_docs['ids']

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_matrix)

    # Group by cluster
    clustered_docs = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_docs:
            clustered_docs[cluster_id] = []
        clustered_docs[cluster_id].append(doc_ids[i])

    # Display
    for cluster_id, docs in clustered_docs.items():
        print(f"\nCluster {cluster_id} ({len(docs)} documents):")
        for doc_id in docs:
            print(f"  - {doc_id}")

    return clustered_docs

# Example
clusters = cluster_documents_by_topic(store, n_clusters=3)
```

### 3. Extract Evidence for Claims

```python
def find_evidence_for_claim(claim: str, store: HierarchicalVectorStore):
    """Find paragraphs that provide evidence for a claim."""
    # Use paragraph-level for evidence extraction
    results = store.query_at_level(claim, 'paragraph', n_results=10)

    evidence_paragraphs = []
    for i, text in enumerate(results['documents'][0]):
        # Check if paragraph contains evidence markers
        evidence_keywords = ['study', 'research', 'data', 'found', 'shows', 'indicates', 'according to']
        has_evidence = any(kw in text.lower() for kw in evidence_keywords)

        if has_evidence:
            meta = results['metadatas'][0][i]
            evidence_paragraphs.append({
                'document_id': meta['document_id'],
                'text': text,
                'similarity': 1 - results['distances'][0][i]
            })

    return evidence_paragraphs

# Example
claim = "Renewable energy reduces carbon emissions"
evidence = find_evidence_for_claim(claim, store)

print(f"Found {len(evidence)} evidence paragraphs:")
for ev in evidence:
    print(f"\n{ev['document_id']} ({ev['similarity']:.2%}):")
    print(f"  {ev['text'][:250]}...")
```

## Best Practices

### 1. Choose Appropriate Level for Query Type

| Query Type | Best Level | Reason |
|------------|------------|--------|
| "Which essays discuss X?" | Document | Broad topic matching |
| "Find main arguments about X" | Section | Logical argument units |
| "Find supporting evidence" | Paragraph | Evidence typically spans paragraphs |
| "Extract exact claims" | Sentence | Precise claim boundaries |

### 2. Store Parent References

```python
# ✅ Good: Can trace back to source
{
    'level': 'paragraph',
    'document_id': 'essay_1',
    'section_index': 2,
    'paragraph_index': 5
}

# ❌ Bad: Orphaned paragraph
{
    'level': 'paragraph',
    'text': '...'
}
```

### 3. Use Consistent Embedding Models

```python
# ✅ Good: Same model for all levels
embedder = HierarchicalEmbedder(model_name='all-MiniLM-L6-v2')

# ❌ Bad: Different models per level (incomparable)
doc_model = SentenceTransformer('all-MiniLM-L6-v2')
sent_model = SentenceTransformer('all-mpnet-base-v2')
```

## Related Documentation

- [NLP Tools Comparison](nlp-tools-comparison.org)
- [Vector Store Integration](nlp-vector-store-integration.md)
- [Document Provenance Tracking](nlp-document-provenance-tracking.md)
- [Semantic Search Strategies](semantic-search-embedding-strategies.md)
- [Hierarchical Document Chunking](hierarchical-document-chunking-strategies.md)
