# Org-roam Zettelkasten Semantic Search Integration

## Overview

Org-roam is an Emacs package that implements the Zettelkasten method for knowledge management using org-mode files. Each note represents a single concept with bidirectional links to related concepts. Integrating semantic search with org-roam requires understanding its unique structure where documents represent concepts rather than traditional hierarchical documents.

## Zettelkasten vs. Traditional Documents

### Traditional Document Structure
```
Book → Chapters → Sections → Paragraphs → Sentences
```

### Zettelkasten Structure
```
Concept Note → Relationship Types → Content Blocks → Sentences
```

**Key Difference**: Each org-roam file is a conceptual unit with explicit relationships to other concepts, not a sequential narrative.

## Org-roam File Structure

### Typical Org-roam Note

```org
:PROPERTIES:
:ID: 20240115103045-machine_learning
:ROAM_ALIASES: "ML" "Machine Learning"
:END:
#+title: Machine Learning Applications

* Definition:
Machine learning is a subset of artificial intelligence...

* Requirements:
- [[id:20240112091523-data_preprocessing][Data Preprocessing]]
- [[id:20240113154032-statistical_methods][Statistical Methods]]

* Outcomes:
- Enables [[id:20240115120345-predictive_models][Predictive Models]]
- Produces [[id:20240116083421-trained_algorithms][Trained Algorithms]]

* Examples:
- Healthcare diagnostics <23>
- Financial forecasting <45>

* RESEARCH Tasks
** RESEARCH [[https://arxiv.org/paper/12345][Read paper on deep learning]]
** IN_PROGRESS Implement neural network example
```

### Key Components

1. **Properties Block**: Unique ID, aliases, metadata
2. **Title**: The concept name
3. **Relationship Sections**: Requirements, Outcomes, Definitions, Examples
4. **Org-roam Links**: `[[id:...][Description]]` - Bidirectional links to other concepts
5. **Bibliography References**: `<23>` style references to external sources
6. **TODO States**: Custom states like RESEARCH, IN_PROGRESS
7. **External Links**: URLs for future processing

## Chunking Strategy for Zettelkasten

### Level 1: Concept-Level (Entire File)
Store the complete concept with all relationships:

```python
{
    "chunk_level": "concept",
    "concept_title": "Machine Learning Applications",
    "concept_id": "20240115103045-machine_learning",
    "aliases": ["ML", "Machine Learning"],
    "text": "[full file content]",
    "org_roam_links": [
        "20240112091523-data_preprocessing",
        "20240113154032-statistical_methods"
    ],
    "backlinks": [
        "20240120092341-artificial_intelligence",
        "20240118143256-neural_networks"
    ]
}
```

**Use Case**: High-level conceptual queries like "Tell me about machine learning"

### Level 2: Relationship-Level (Section Headers)
Each relationship type becomes a searchable chunk:

```python
{
    "chunk_level": "relationship",
    "relationship_type": "requirement",
    "concept_title": "Machine Learning Applications",
    "text": "Data Preprocessing and Statistical Methods are required...",
    "related_concepts": [
        "20240112091523-data_preprocessing",
        "20240113154032-statistical_methods"
    ]
}
```

**Use Case**: Queries about relationships like "What are the requirements for machine learning?"

### Level 3: Content-Block Level (Paragraphs)
Individual paragraphs within sections:

```python
{
    "chunk_level": "content_block",
    "relationship_type": "definition",
    "concept_title": "Machine Learning Applications",
    "hierarchy_path": "Definition/Paragraph 1",
    "text": "Machine learning is a subset of artificial intelligence..."
}
```

**Use Case**: Specific facts within a concept's definition

### Level 4: Sentence-Level
For precise fact retrieval with coreference resolution applied.

## Handling Org-roam Links

### Link Types

**ID-based Links** (Recommended by org-roam):
```org
[[id:20240115103045-machine_learning][Machine Learning]]
```

**File-based Links** (Legacy):
```org
[[file:20240115103045-machine_learning.org][Machine Learning]]
```

### Link Extraction

```python
import re
import orgparse

def extract_org_roam_links(org_file_path):
    """Extract all org-roam links from a file."""

    node = orgparse.load(org_file_path)

    # ID-based links
    id_pattern = r'\[\[id:([^\]]+)\]\[([^\]]*)\]\]'
    id_links = re.findall(id_pattern, node.get_body())

    # File-based links
    file_pattern = r'\[\[file:([^\]]+\.org)\]\[([^\]]*)\]\]'
    file_links = re.findall(file_pattern, node.get_body())

    return {
        "id_links": [(link_id, description) for link_id, description in id_links],
        "file_links": [(filename, description) for filename, description in file_links]
    }
```

### Bidirectional Link Support

Org-roam maintains a database of backlinks:

```python
def get_concept_with_links(concept_id):
    """Get concept with forward and backlinks."""

    # Forward links (links from this concept to others)
    forward_links = extract_org_roam_links(f"{concept_id}.org")

    # Backlinks (other concepts linking to this one)
    # Query org-roam database or grep through all files
    backlinks = find_backlinks(concept_id)

    return {
        "concept_id": concept_id,
        "forward_links": forward_links,
        "backlinks": backlinks
    }
```

### Link Context for Embeddings

Include linked concept titles in embeddings for better semantic search:

```python
def enrich_concept_for_embedding(concept_text, links):
    """Add link context to concept text."""

    linked_concepts = [get_concept_title(link_id) for link_id in links]

    enriched = f"{concept_text}\n\nRelated concepts: {', '.join(linked_concepts)}"

    return enriched
```

## Bibliography Reference System

### Current Format
```org
Healthcare diagnostics <23> shows promising results.
```

Where `<23>` references an entry in a bibliography org-mode table.

### Reference Extraction

```python
def extract_bibliography_refs(org_file_path):
    """Extract bibliography reference markers."""

    with open(org_file_path, 'r') as f:
        content = f.read()

    ref_pattern = r'<(\d+)>'
    refs = re.findall(ref_pattern, content)

    return list(set(refs))  # Unique references
```

### Reference Metadata

Store references alongside chunks:

```python
{
    "text": "Healthcare diagnostics shows promising results.",
    "bibliography_refs": ["<23>"],
    "sources": [
        {
            "ref_id": "23",
            "title": "Medical AI Applications",
            "authors": "Smith et al.",
            "year": 2023,
            "url": "https://journal.example/paper"
        }
    ]
}
```

## TODO States and Research Links

### Custom TODO States

Org-roam notes often have custom TODO states:
- **RESEARCH**: Links/topics to investigate
- **IN_PROGRESS**: Active work
- **COMPLETED**: Finished tasks

### Extracting Research TODOs

```python
def extract_research_todos(org_file_path):
    """Extract RESEARCH state TODOs with URLs."""

    node = orgparse.load(org_file_path)

    research_items = []
    for heading in node[1:]:  # Skip root node
        if heading.todo == "RESEARCH":
            # Extract URLs from heading
            url_pattern = r'\[\[([^]]+)\]\[([^]]*)\]\]'
            urls = re.findall(url_pattern, heading.get_body())

            research_items.append({
                "heading": heading.heading,
                "urls": urls,
                "priority": heading.priority
            })

    return research_items
```

### Metadata for Incomplete Knowledge

Flag concepts with pending research:

```python
{
    "concept_title": "Machine Learning Applications",
    "has_pending_research": True,
    "research_count": 3,
    "research_topics": [
        "Deep learning applications",
        "Neural network architectures",
        "Transfer learning methods"
    ]
}
```

## Knowledge Graph Integration

### Graph Structure

```python
knowledge_graph = {
    "nodes": [
        {
            "id": "20240115103045-machine_learning",
            "title": "Machine Learning Applications",
            "type": "concept"
        }
    ],
    "edges": [
        {
            "source": "20240115103045-machine_learning",
            "target": "20240112091523-data_preprocessing",
            "relationship": "requires",
            "weight": 1.0
        }
    ]
}
```

### Graph-Enhanced Search

Use graph structure to improve search relevance:

```python
def graph_enhanced_search(query, initial_results):
    """Re-rank results based on concept relationships."""

    # Get concepts from initial results
    concept_ids = [r["concept_id"] for r in initial_results]

    # Find highly connected concepts
    graph_scores = compute_centrality(knowledge_graph, concept_ids)

    # Combine vector similarity with graph centrality
    for result in initial_results:
        concept_id = result["concept_id"]
        result["final_score"] = (
            0.7 * result["similarity_score"] +
            0.3 * graph_scores[concept_id]
        )

    return sorted(initial_results, key=lambda x: x["final_score"], reverse=True)
```

## Query Strategies for Zettelkasten

### Concept Discovery
"What do I know about machine learning?"
- Search at concept level
- Return full concept with relationships

### Relationship Queries
"What are the prerequisites for machine learning?"
- Filter by relationship_type = "requirement"
- Search within relationship sections

### Cross-Concept Queries
"How does machine learning relate to statistical methods?"
- Search both concepts
- Analyze org-roam links between them
- Return relationship path

### Source-Grounded Queries
"What did Smith et al. say about healthcare AI?"
- Filter by bibliography_refs
- Return chunks with specific references

## ChromaDB Schema for Zettelkasten

```python
{
    # Identity
    "concept_id": "20240115103045-machine_learning",
    "concept_title": "Machine Learning Applications",
    "concept_aliases": ["ML", "Machine Learning"],

    # Chunk Information
    "chunk_level": "concept|relationship|content_block|sentence",
    "relationship_type": "requirement|outcome|definition|example|null",

    # Links
    "org_roam_links": ["id1", "id2", "id3"],
    "backlinks": ["id4", "id5"],
    "link_count": 5,
    "backlink_count": 2,

    # References
    "bibliography_refs": ["<23>", "<45>"],
    "external_urls": ["https://example.com/article"],

    # Status
    "todo_states": ["RESEARCH", "IN_PROGRESS"],
    "has_pending_research": True,
    "research_count": 3,

    # Content
    "hierarchy_path": "Requirements/Data Preprocessing",
    "text": "[actual content]",

    # Processing
    "coreference_resolved": True,
    "confidence_score": 0.89,
    "processing_date": "2024-01-15T10:30:00Z"
}
```

## Implementation Considerations

### Org-roam Database
Org-roam maintains its own SQLite database with node and link information. Consider:
- Querying org-roam database directly for link information
- Syncing with vector database when notes are modified
- Using org-roam hooks to trigger re-processing

### File Watching
Monitor org-roam directory for changes:

```python
import watchdog

def on_org_file_modified(file_path):
    """Re-process org-roam file when modified."""

    # Extract concept ID from filename
    concept_id = extract_concept_id(file_path)

    # Reprocess and update embeddings
    process_and_update_concept(concept_id, file_path)
```

### Incremental Updates
Only reprocess changed concepts and their directly linked neighbors to maintain graph coherence.

## Related Topics

- **Chunking Strategies**: See hierarchical-document-chunking-strategies.md
- **ChromaDB Implementation**: See chromadb-hierarchical-storage.md
- **Coreference Resolution**: See coreference-resolution-nlp.md
- **Multi-Format Processing**: See multi-format-document-processing.md
