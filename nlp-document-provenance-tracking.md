# NLP Document Provenance Tracking

## Overview

This document explains how to maintain the connection between source documents and NLP analysis results across multiple iterations, particularly when analyses happen weeks or months apart.

## Core Challenge

When analyzing markdown documents with spaCy or other NLP pipelines:

1. **Initial Analysis**: Process document → generate entities, embeddings, arguments
2. **Storage**: Save analysis results somewhere persistent
3. **Time Gap**: Days, weeks, or months pass
4. **Re-Analysis**: Need to re-process same document with refined pipeline
5. **Association Problem**: How to link new results to old results and original document?

## Solution Approaches

### 1. Content-Based Document IDs (Recommended)

Use content hashing to generate stable document identifiers that survive file moves and renames.

#### Implementation

```python
import hashlib
from pathlib import Path

def generate_document_id(filepath: str) -> str:
    """Generate stable ID from document content."""
    content = Path(filepath).read_text(encoding='utf-8')
    return hashlib.sha256(content.encode()).hexdigest()[:16]

# Example usage
doc_id = generate_document_id("student_essay_1.md")
# => "a3f2d9c8e1b4f7a2"
```

**Advantages**:
- Survives file renames and moves
- Same content always produces same ID
- Detects document changes automatically

**Disadvantages**:
- ID changes if content is edited (even minor changes)
- Need separate versioning strategy for edited documents

### 2. Path-Based IDs with Change Detection

Use file path as primary ID, detect changes via content hash.

```python
import hashlib
from pathlib import Path
from datetime import datetime

def generate_document_metadata(filepath: str) -> dict:
    """Generate metadata including path and content hash."""
    path = Path(filepath)
    content = path.read_text(encoding='utf-8')
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    return {
        'document_id': str(path.absolute()),
        'content_hash': content_hash,
        'filename': path.name,
        'analyzed_at': datetime.utcnow().isoformat(),
        'file_size': path.stat().st_size,
        'modified_at': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    }
```

**Advantages**:
- Human-readable IDs (file paths)
- Can track document edits via hash changes
- Works well with version control systems

**Disadvantages**:
- Breaks if files are moved/renamed
- Need additional logic to handle path changes

### 3. Hybrid Approach (Best for Most Use Cases)

Combine path-based primary ID with content hashing for version tracking.

```python
import hashlib
import json
from pathlib import Path
from datetime import datetime

class DocumentRegistry:
    """Track documents and their analysis history."""

    def __init__(self, registry_path: str = "document_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}

    def _save_registry(self):
        """Persist registry to disk."""
        self.registry_path.write_text(
            json.dumps(self.registry, indent=2)
        )

    def register_document(self, filepath: str) -> dict:
        """Register document and detect if it's new or changed."""
        path = Path(filepath).absolute()
        doc_id = str(path)
        content = path.read_text(encoding='utf-8')
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        metadata = {
            'document_id': doc_id,
            'filepath': str(path),
            'content_hash': content_hash,
            'filename': path.name,
            'registered_at': datetime.utcnow().isoformat()
        }

        # Check if document is known
        if doc_id in self.registry:
            old_hash = self.registry[doc_id]['content_hash']
            if old_hash != content_hash:
                metadata['status'] = 'modified'
                metadata['previous_hash'] = old_hash
                metadata['version'] = self.registry[doc_id].get('version', 1) + 1
            else:
                metadata['status'] = 'unchanged'
                metadata['version'] = self.registry[doc_id].get('version', 1)
        else:
            metadata['status'] = 'new'
            metadata['version'] = 1

        # Update registry
        self.registry[doc_id] = metadata
        self._save_registry()

        return metadata

# Example usage
registry = DocumentRegistry()
doc_info = registry.register_document("student_essay_1.md")

if doc_info['status'] == 'new':
    print(f"First time analyzing: {doc_info['filename']}")
elif doc_info['status'] == 'modified':
    print(f"Document changed since v{doc_info['version']-1}")
elif doc_info['status'] == 'unchanged':
    print(f"Document unchanged, version {doc_info['version']}")
```

## Storing Analysis Results

### Option 1: JSON Files (Simple, Good for Development)

Store results alongside source documents or in dedicated output directory.

```python
import json
from pathlib import Path
from datetime import datetime

def save_analysis_results(document_id: str, analysis: dict, output_dir: str = "analysis_results"):
    """Save analysis results as JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create filename from document_id and timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{document_id}_{timestamp}.json"

    result_file = output_path / filename
    result_file.write_text(json.dumps(analysis, indent=2))

    return str(result_file)

# Example usage
analysis = {
    'document_id': 'a3f2d9c8e1b4f7a2',
    'iteration': 1,
    'entities': [{'text': 'climate change', 'label': 'TOPIC'}],
    'arguments': ['We need action on climate change'],
    'embedding': [0.123, 0.456, ...]  # 384-dim vector
}

save_analysis_results(analysis['document_id'], analysis)
```

**Directory Structure**:
```
project/
├── documents/
│   ├── student_essay_1.md
│   └── student_essay_2.md
├── analysis_results/
│   ├── a3f2d9c8e1b4f7a2_20250108_143022.json
│   ├── a3f2d9c8e1b4f7a2_20250115_091545.json  # Iteration 2
│   └── b7e9f3a1c6d8e2b5_20250108_143045.json
└── document_registry.json
```

### Option 2: SQLite Database (Better for Queries)

Use relational database for structured storage and efficient querying.

```python
import sqlite3
import json
from datetime import datetime

class AnalysisDatabase:
    """Store and retrieve analysis results."""

    def __init__(self, db_path: str = "analysis.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                filename TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                version INTEGER DEFAULT 1
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL,
                pipeline_version TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS arguments (
                argument_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                claim TEXT NOT NULL,
                evidence TEXT,
                stance TEXT,
                confidence REAL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                level TEXT NOT NULL,  -- 'document', 'section', 'paragraph'
                vector BLOB NOT NULL,  -- numpy array serialized
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id)
            )
        """)

        self.conn.commit()

    def store_analysis(self, document_id: str, iteration: int,
                      entities: list, arguments: list, embedding: list,
                      pipeline_version: str = "1.0") -> int:
        """Store complete analysis results."""
        # Insert analysis record
        cursor = self.conn.execute("""
            INSERT INTO analyses (document_id, iteration, analyzed_at, pipeline_version)
            VALUES (?, ?, ?, ?)
        """, (document_id, iteration, datetime.utcnow().isoformat(), pipeline_version))

        analysis_id = cursor.lastrowid

        # Insert entities
        for entity in entities:
            self.conn.execute("""
                INSERT INTO entities (analysis_id, text, label, start_char, end_char)
                VALUES (?, ?, ?, ?, ?)
            """, (analysis_id, entity['text'], entity['label'],
                  entity.get('start'), entity.get('end')))

        # Insert arguments
        for arg in arguments:
            self.conn.execute("""
                INSERT INTO arguments (analysis_id, claim, evidence, stance, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (analysis_id, arg.get('claim'), arg.get('evidence'),
                  arg.get('stance'), arg.get('confidence')))

        # Insert embedding
        import numpy as np
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        self.conn.execute("""
            INSERT INTO embeddings (analysis_id, level, vector)
            VALUES (?, ?, ?)
        """, (analysis_id, 'document', embedding_bytes))

        self.conn.commit()
        return analysis_id

    def get_latest_analysis(self, document_id: str) -> dict:
        """Retrieve most recent analysis for document."""
        cursor = self.conn.execute("""
            SELECT analysis_id, iteration, analyzed_at, pipeline_version
            FROM analyses
            WHERE document_id = ?
            ORDER BY iteration DESC
            LIMIT 1
        """, (document_id,))

        row = cursor.fetchone()
        if not row:
            return None

        analysis_id, iteration, analyzed_at, pipeline_version = row

        # Fetch entities
        entities = []
        for row in self.conn.execute("""
            SELECT text, label, start_char, end_char
            FROM entities WHERE analysis_id = ?
        """, (analysis_id,)):
            entities.append({
                'text': row[0],
                'label': row[1],
                'start': row[2],
                'end': row[3]
            })

        # Fetch arguments
        arguments = []
        for row in self.conn.execute("""
            SELECT claim, evidence, stance, confidence
            FROM arguments WHERE analysis_id = ?
        """, (analysis_id,)):
            arguments.append({
                'claim': row[0],
                'evidence': row[1],
                'stance': row[2],
                'confidence': row[3]
            })

        # Fetch embedding
        cursor = self.conn.execute("""
            SELECT vector FROM embeddings
            WHERE analysis_id = ? AND level = 'document'
        """, (analysis_id,))
        embedding_row = cursor.fetchone()

        import numpy as np
        embedding = None
        if embedding_row:
            embedding = np.frombuffer(embedding_row[0], dtype=np.float32).tolist()

        return {
            'analysis_id': analysis_id,
            'document_id': document_id,
            'iteration': iteration,
            'analyzed_at': analyzed_at,
            'pipeline_version': pipeline_version,
            'entities': entities,
            'arguments': arguments,
            'embedding': embedding
        }

    def get_all_iterations(self, document_id: str) -> list:
        """Get all analysis iterations for comparison."""
        cursor = self.conn.execute("""
            SELECT analysis_id, iteration, analyzed_at, pipeline_version
            FROM analyses
            WHERE document_id = ?
            ORDER BY iteration ASC
        """, (document_id,))

        return [
            {
                'analysis_id': row[0],
                'iteration': row[1],
                'analyzed_at': row[2],
                'pipeline_version': row[3]
            }
            for row in cursor.fetchall()
        ]

# Example usage
db = AnalysisDatabase()

# Store first iteration
analysis_id = db.store_analysis(
    document_id='a3f2d9c8e1b4f7a2',
    iteration=1,
    entities=[{'text': 'climate change', 'label': 'TOPIC', 'start': 10, 'end': 24}],
    arguments=[{'claim': 'Action needed on climate', 'stance': 'support', 'confidence': 0.92}],
    embedding=[0.123, 0.456, ...],
    pipeline_version='1.0'
)

# Later: retrieve for comparison
latest = db.get_latest_analysis('a3f2d9c8e1b4f7a2')
print(f"Found iteration {latest['iteration']} from {latest['analyzed_at']}")

# View iteration history
iterations = db.get_all_iterations('a3f2d9c8e1b4f7a2')
for it in iterations:
    print(f"Iteration {it['iteration']}: {it['analyzed_at']} (pipeline v{it['pipeline_version']})")
```

### Option 3: Vector Database (Best for Similarity Search)

Use ChromaDB or similar for storing embeddings with metadata.

```python
import chromadb
from chromadb.config import Settings

class VectorStoreWithProvenance:
    """Store embeddings with full document provenance."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet"
        ))
        self.collection = self.client.get_or_create_collection(
            name="document_analyses",
            metadata={"hnsw:space": "cosine"}
        )

    def store_analysis(self, document_id: str, iteration: int,
                      content: str, embedding: list, metadata: dict):
        """Store document analysis with provenance."""
        # Create unique ID for this iteration
        unique_id = f"{document_id}_iter_{iteration}"

        # Merge provenance metadata
        full_metadata = {
            'document_id': document_id,
            'iteration': iteration,
            'analyzed_at': metadata.get('analyzed_at'),
            'pipeline_version': metadata.get('pipeline_version', '1.0'),
            'filepath': metadata.get('filepath'),
            'content_hash': metadata.get('content_hash'),
            'entity_count': metadata.get('entity_count', 0),
            'argument_count': metadata.get('argument_count', 0)
        }

        self.collection.add(
            ids=[unique_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[full_metadata]
        )

    def get_document_history(self, document_id: str) -> list:
        """Retrieve all iterations for a document."""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["embeddings", "metadatas", "documents"]
        )

        # Sort by iteration
        iterations = []
        for i, meta in enumerate(results['metadatas']):
            iterations.append({
                'id': results['ids'][i],
                'iteration': meta['iteration'],
                'analyzed_at': meta['analyzed_at'],
                'embedding': results['embeddings'][i],
                'document': results['documents'][i],
                'metadata': meta
            })

        return sorted(iterations, key=lambda x: x['iteration'])

    def compare_iterations(self, document_id: str, iter1: int, iter2: int) -> dict:
        """Compare two analysis iterations."""
        history = self.get_document_history(document_id)

        v1 = next((h for h in history if h['iteration'] == iter1), None)
        v2 = next((h for h in history if h['iteration'] == iter2), None)

        if not v1 or not v2:
            return None

        # Calculate embedding similarity
        import numpy as np
        emb1 = np.array(v1['embedding'])
        emb2 = np.array(v2['embedding'])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return {
            'document_id': document_id,
            'iteration_1': iter1,
            'iteration_2': iter2,
            'embedding_similarity': float(similarity),
            'entity_count_change': v2['metadata']['entity_count'] - v1['metadata']['entity_count'],
            'argument_count_change': v2['metadata']['argument_count'] - v1['metadata']['argument_count'],
            'time_between_analyses': v2['analyzed_at']  # Would need datetime parsing
        }

# Example usage
vector_store = VectorStoreWithProvenance()

# Store iteration 1
vector_store.store_analysis(
    document_id='a3f2d9c8e1b4f7a2',
    iteration=1,
    content="Student essay about climate change...",
    embedding=[0.123, 0.456, ...],
    metadata={
        'analyzed_at': '2025-01-08T14:30:22',
        'pipeline_version': '1.0',
        'filepath': '/docs/student_essay_1.md',
        'content_hash': 'a3f2d9c8',
        'entity_count': 15,
        'argument_count': 3
    }
)

# Later: retrieve history
history = vector_store.get_document_history('a3f2d9c8e1b4f7a2')
print(f"Document has {len(history)} analysis iterations")

# Compare iterations
comparison = vector_store.compare_iterations('a3f2d9c8e1b4f7a2', 1, 2)
print(f"Embedding similarity: {comparison['embedding_similarity']:.2%}")
```

## Handling Long Time Gaps

### Strategy 1: Pipeline Versioning

Track which version of pipeline was used for each analysis.

```python
class PipelineVersion:
    """Track NLP pipeline configuration."""

    def __init__(self, version: str, config: dict):
        self.version = version
        self.config = config

    def get_info(self) -> dict:
        return {
            'version': self.version,
            'spacy_model': self.config.get('spacy_model'),
            'sbert_model': self.config.get('sbert_model'),
            'argument_extractor': self.config.get('argument_extractor'),
            'parameters': self.config.get('parameters', {})
        }

# Version 1.0 (Initial exploration)
pipeline_v1 = PipelineVersion('1.0', {
    'spacy_model': 'en_core_web_sm',
    'sbert_model': 'all-MiniLM-L6-v2',
    'argument_extractor': 'rule_based',
    'parameters': {
        'similarity_threshold': 0.7
    }
})

# Version 2.0 (After discovering new features)
pipeline_v2 = PipelineVersion('2.0', {
    'spacy_model': 'en_core_web_lg',  # Upgraded
    'sbert_model': 'all-mpnet-base-v2',  # Better embeddings
    'argument_extractor': 'hybrid',  # Improved
    'parameters': {
        'similarity_threshold': 0.85,  # Adjusted
        'claim_confidence_min': 0.8  # New parameter
    }
})

# Store pipeline version with each analysis
def analyze_with_version(document_id: str, pipeline_version: PipelineVersion) -> dict:
    """Run analysis and record pipeline version."""
    # ... run analysis ...

    return {
        'document_id': document_id,
        'pipeline_version': pipeline_version.version,
        'pipeline_config': pipeline_version.get_info(),
        'results': analysis_results
    }
```

### Strategy 2: Backwards Compatibility Checks

Detect when re-analysis is needed due to pipeline changes.

```python
def should_reanalyze(document_id: str, current_pipeline: str) -> bool:
    """Check if document needs re-analysis with new pipeline."""
    db = AnalysisDatabase()
    latest = db.get_latest_analysis(document_id)

    if not latest:
        return True  # Never analyzed

    if latest['pipeline_version'] != current_pipeline:
        return True  # Pipeline upgraded

    # Check if document content changed
    registry = DocumentRegistry()
    doc_info = registry.register_document(document_id)

    if doc_info['status'] == 'modified':
        return True  # Document edited

    return False  # Up to date

# Example workflow
current_pipeline = '2.0'
document = 'student_essay_1.md'

if should_reanalyze(document, current_pipeline):
    print(f"Re-analyzing {document} with pipeline v{current_pipeline}")
    # ... run analysis ...
else:
    print(f"Using cached results for {document}")
    # ... load from database ...
```

## Complete Workflow Example

Putting it all together with full provenance tracking:

```python
import hashlib
from pathlib import Path
from datetime import datetime
import spacy
from sentence_transformers import SentenceTransformer

class ProvenanceTrackedAnalysis:
    """Complete NLP analysis system with provenance tracking."""

    def __init__(self, pipeline_version: str = "1.0"):
        self.pipeline_version = pipeline_version
        self.nlp = spacy.load("en_core_web_sm")
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.registry = DocumentRegistry()
        self.db = AnalysisDatabase()
        self.vector_store = VectorStoreWithProvenance()

    def analyze_document(self, filepath: str, force_rerun: bool = False) -> dict:
        """Analyze document with full provenance tracking."""
        # Register document
        doc_info = self.registry.register_document(filepath)
        document_id = doc_info['document_id']

        # Check if re-analysis needed
        if not force_rerun:
            latest = self.db.get_latest_analysis(document_id)
            if latest and latest['pipeline_version'] == self.pipeline_version:
                if doc_info['status'] == 'unchanged':
                    print(f"Using cached analysis (iteration {latest['iteration']})")
                    return latest

        # Determine iteration number
        iterations = self.db.get_all_iterations(document_id)
        iteration = len(iterations) + 1

        # Load and process document
        content = Path(filepath).read_text(encoding='utf-8')
        doc = self.nlp(content)

        # Extract features
        entities = [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            for ent in doc.ents
        ]

        # Extract arguments (simplified)
        arguments = self._extract_arguments(doc)

        # Generate embedding
        embedding = self.sbert.encode(content).tolist()

        # Store in database
        analysis_id = self.db.store_analysis(
            document_id=document_id,
            iteration=iteration,
            entities=entities,
            arguments=arguments,
            embedding=embedding,
            pipeline_version=self.pipeline_version
        )

        # Store in vector database
        self.vector_store.store_analysis(
            document_id=document_id,
            iteration=iteration,
            content=content,
            embedding=embedding,
            metadata={
                'analyzed_at': datetime.utcnow().isoformat(),
                'pipeline_version': self.pipeline_version,
                'filepath': filepath,
                'content_hash': doc_info['content_hash'],
                'entity_count': len(entities),
                'argument_count': len(arguments)
            }
        )

        print(f"✓ Analyzed {doc_info['filename']} (iteration {iteration})")

        return {
            'analysis_id': analysis_id,
            'document_id': document_id,
            'iteration': iteration,
            'pipeline_version': self.pipeline_version,
            'entities': entities,
            'arguments': arguments,
            'embedding': embedding,
            'provenance': doc_info
        }

    def _extract_arguments(self, doc) -> list:
        """Extract arguments from document (placeholder)."""
        # Simplified argument extraction
        arguments = []
        for sent in doc.sents:
            if any(token.text.lower() in ['argue', 'claim', 'believe'] for token in sent):
                arguments.append({
                    'claim': sent.text,
                    'stance': 'unknown',
                    'confidence': 0.5
                })
        return arguments

    def compare_document_versions(self, filepath: str) -> dict:
        """Compare all analysis iterations for a document."""
        doc_info = self.registry.register_document(filepath)
        document_id = doc_info['document_id']

        iterations = self.db.get_all_iterations(document_id)

        if len(iterations) < 2:
            return {'message': 'Need at least 2 iterations to compare'}

        comparisons = []
        for i in range(len(iterations) - 1):
            iter1 = iterations[i]['iteration']
            iter2 = iterations[i + 1]['iteration']
            comp = self.vector_store.compare_iterations(document_id, iter1, iter2)
            comparisons.append(comp)

        return {
            'document_id': document_id,
            'total_iterations': len(iterations),
            'comparisons': comparisons
        }

# Example usage workflow
analyzer = ProvenanceTrackedAnalysis(pipeline_version="1.0")

# Initial analysis (Iteration 1)
result1 = analyzer.analyze_document("student_essay_1.md")
print(f"Found {len(result1['entities'])} entities, {len(result1['arguments'])} arguments")

# ... weeks later, no changes to document ...
result2 = analyzer.analyze_document("student_essay_1.md")
# Output: "Using cached analysis (iteration 1)"

# ... student edits essay ...
result3 = analyzer.analyze_document("student_essay_1.md")
# Output: "✓ Analyzed student_essay_1.md (iteration 2)"

# Compare all versions
comparison = analyzer.compare_document_versions("student_essay_1.md")
print(f"Document has {comparison['total_iterations']} analysis iterations")
for comp in comparison['comparisons']:
    print(f"  v{comp['iteration_1']} → v{comp['iteration_2']}: "
          f"{comp['embedding_similarity']:.2%} similar, "
          f"{comp['entity_count_change']:+d} entities, "
          f"{comp['argument_count_change']:+d} arguments")
```

## Best Practices

### 1. Always Generate Document IDs from Content

```python
# ✅ Good: Content-based ID
doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]

# ❌ Bad: Random ID (won't survive re-runs)
doc_id = str(uuid.uuid4())
```

### 2. Store Iteration Number with Every Analysis

```python
# ✅ Good: Explicit iteration tracking
analysis = {
    'document_id': doc_id,
    'iteration': 3,
    'analyzed_at': timestamp,
    # ...
}

# ❌ Bad: No iteration tracking (can't compare versions)
analysis = {
    'document_id': doc_id,
    # Missing iteration info
}
```

### 3. Record Pipeline Version

```python
# ✅ Good: Know which pipeline produced results
'pipeline_version': '2.0',
'pipeline_config': {
    'spacy_model': 'en_core_web_lg',
    'similarity_threshold': 0.85
}

# ❌ Bad: Can't reproduce or understand results later
# (no version info)
```

### 4. Use Separate Storage for Different Data Types

- **Structured data** (entities, arguments) → SQLite
- **Embeddings** → ChromaDB / FAISS
- **Full documents** → Filesystem
- **Metadata** → JSON registry

### 5. Plan for Scale

```python
# ✅ Good: Indexed database queries
CREATE INDEX idx_document_analyses ON analyses(document_id, iteration);

# ✅ Good: Batch processing
for batch in chunks(documents, size=100):
    process_batch(batch)

# ❌ Bad: Loading everything into memory
all_results = [analyze(doc) for doc in all_documents]  # OOM risk
```

## Markdown-Specific Considerations

### Preserving Document Structure

When processing markdown, maintain connection to original structure:

```python
def parse_markdown_with_positions(filepath: str) -> dict:
    """Parse markdown preserving section boundaries."""
    content = Path(filepath).read_text(encoding='utf-8')

    sections = []
    current_section = {'heading': None, 'content': [], 'start': 0}

    for i, line in enumerate(content.split('\n')):
        if line.startswith('#'):
            # New section
            if current_section['content']:
                current_section['end'] = i
                sections.append(current_section)

            current_section = {
                'heading': line.strip('# '),
                'content': [],
                'start': i,
                'level': len(line) - len(line.lstrip('#'))
            }
        else:
            current_section['content'].append(line)

    # Add final section
    if current_section['content']:
        current_section['end'] = len(content.split('\n'))
        sections.append(current_section)

    return {
        'filepath': filepath,
        'full_content': content,
        'sections': sections,
        'content_hash': hashlib.sha256(content.encode()).hexdigest()
    }

# Use section boundaries for provenance
def analyze_markdown_sections(filepath: str) -> list:
    """Analyze each markdown section separately."""
    parsed = parse_markdown_with_positions(filepath)
    section_analyses = []

    for section in parsed['sections']:
        section_content = '\n'.join(section['content'])
        doc = nlp(section_content)

        section_analyses.append({
            'heading': section['heading'],
            'line_start': section['start'],
            'line_end': section['end'],
            'entities': [{'text': ent.text, 'label': ent.label_} for ent in doc.ents],
            # Can now link entities back to exact markdown location!
        })

    return section_analyses
```

## Summary

**Key Principles**:

1. **Content hashing** for stable document IDs
2. **Iteration tracking** for version history
3. **Pipeline versioning** for reproducibility
4. **Multi-store architecture** for different data types
5. **Metadata preservation** linking results to source

**Recommended Stack**:
- Document registry: JSON file
- Structured data: SQLite
- Embeddings: ChromaDB
- Full text: Filesystem (markdown files)

This approach ensures you can:
- Re-analyze documents months later
- Compare iteration results
- Trace any entity/argument back to source
- Handle document edits gracefully
- Reproduce analyses with same pipeline version
