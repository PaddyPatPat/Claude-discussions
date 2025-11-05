# Semantic Document Processing System - Project Specification

## Project Overview

This project aims to build a sophisticated semantic document processing and search system for a personal knowledge management library. The system will implement hierarchical chunking, coreference resolution, and multi-level embedding storage using ChromaDB as the vector database backend.

## Core Requirements

### Primary Objectives
- Implement recursive document splitting with hierarchical chunking
- Maintain context through coreference resolution
- Support multiple document formats and domains
- Enable multi-granularity semantic search
- Preserve document relationships and cross-references

### Key Features
- **Hierarchical Document Chunking**: Split documents recursively into chapters → sections → paragraphs → sentences
- **Context Preservation**: Resolve pronouns and references before chunking (e.g., "He walks" → "Dave walks")
- **Multi-Level Embeddings**: Store embeddings at different granularity levels for different query types
- **Cross-Document Relationships**: Maintain and search across document references and links
- **Zettelkasten Integration**: Special handling for org-roam knowledge graph structure

## Document Types and Formats

### Supported Formats
- **Org-mode files** (primary format for Zettelkasten)
- **Markdown files**
- **PDF documents**
- **HTML files**
- **EPUB books**

### Content Domains
- **Books**
  - Application manuals
  - Non-fiction books
- **Academic papers**
- **Web articles**
- **Web comment threads** (including nested comments)
- **To-do lists**
- **Org-roam Zettelkasten** (hundreds to thousands of concept files)

## Org-Roam Zettelkasten Specifications

### Structure Requirements
- Standard org-mode hierarchies using headlines (*, **, ***)
- Custom TODO states: "RESEARCH", "IN PROGRESS" (in addition to standard states)
- Special relationship headings: "* Requirement:", "* Outcome:"
- Org-roam links between concept documents
- Bibliography references using format: "<23>"
- Research URLs stored in RESEARCH TODO states

### Cross-Document Relationships
- **Org-roam links**: `[[file:other.org][description]]` format
- **Bibliography table**: Org-mode table with references matching "<23>" format
- **Website links**: URLs in RESEARCH todos (for future processing)
- **Backlinks**: Automatic org-roam generated backlinks

## Query Patterns and Use Cases

### Query Types
- **Fact-finding queries**: "What is the temperature threshold for polymer degradation?"
- **Conceptual queries**: "How does machine learning apply to healthcare?"
- **Comparative queries**: "What are the differences between approach A and B?"
- **Procedural queries**: "How do I configure the authentication system?"

### Expected Response Granularity
- High-level summaries with drill-down capabilities
- Clear relationships between concepts
- Hierarchical context information ("From Chapter 3, Section 2.1 of Book X")

## Technical Architecture

### Chunking Strategy

#### Multi-Level Hierarchy
```
Level 1: Document/Chapter level (2000-4000 tokens)
Level 2: Section level (800-1500 tokens)
Level 3: Paragraph level (200-500 tokens)
Level 4: Sentence level (20-100 tokens)
```

#### Zettelkasten-Specific Chunking
```
Level 1: Concept-level (entire org-roam file)
Level 2: Relationship-level ("Requirements:", "Outcomes:", etc.)
Level 3: Content-block level (paragraphs within sections)
Level 4: Sentence-level (for precise fact retrieval)
```

### Tool Stack

#### Core Processing Pipeline
- **Document parsing**: `pypandoc`, `orgparse`, `ebooklib`
- **Structure detection**: Custom parsers per format + `spaCy`
- **Coreference resolution**: `spaCy` with `neuralcoref` or `AllenNLP`
- **Embeddings**: `sentence-transformers` (multiple models for different chunk sizes)
- **Storage**: ChromaDB with hierarchical metadata

#### Embedding Models
- **Large chunks**: `all-MiniLM-L12-v2` or `all-mpnet-base-v2`
- **Small chunks**: `all-MiniLM-L6-v2` for speed
- **Specialized**: Consider `SciBERT` for academic papers

### Processing Pipeline Order
1. Format-specific structure extraction
2. Hierarchical chunking (preserve parent-child relationships)
3. Coreference resolution per chunk level
4. Multi-level embedding generation
5. Storage with rich metadata

## Performance Requirements

### Processing Timeline
- **Batch processing acceptable** (not real-time required)
- Target: Process one book per day
- Incremental processing preferred for changed documents

### Query Response Time
- **Live user queries**: A few seconds acceptable
- **Automated process queries**: Minutes acceptable

### Scale Considerations
- **Documents**: Tens of thousands of articles, hundreds of books
- **Org-roam files**: Hundreds to thousands of concept files
- **Users**: Single user system (outside of batch processing)
- **Hardware**: Home lab with Nvidia workstation GPUs and Apple Silicon

## Data Storage Schema

### ChromaDB Metadata Structure
```python
{
    "doc_id": "unique_doc_identifier",
    "doc_type": "org-mode|pdf|html|epub|markdown",
    "chunk_level": "document|chapter|section|paragraph|sentence",
    "hierarchy_path": "Chapter 1/Section 1.1/Paragraph 3",
    "parent_chunk_id": "parent_reference",
    "child_chunk_ids": ["child1", "child2"],
    "confidence_score": 0.85,
    "processing_notes": "coreference_resolved, manual_review_needed"
}
```

### Zettelkasten-Specific Metadata
```python
{
    "concept_title": "Machine Learning Applications",
    "relationship_type": "requirement|outcome|definition|example",
    "org_roam_links": ["concept2.org", "concept5.org"],
    "bibliography_refs": ["<23>", "<45>"],
    "todo_states": ["RESEARCH", "IN_PROGRESS"],
    "backlinks": ["concept7.org", "concept12.org"]
}
```

### Comment Thread Metadata
```python
{
    "thread_id": "reddit_post_123",
    "comment_id": "comment_456",
    "parent_comment_id": "comment_455",
    "depth_level": 7,
    "timestamp": "2024-01-15T10:30:00Z",
    "author": "username",
    "thread_context": "Parent discussion about X topic"
}
```

## Quality and Error Handling

### Quality Control Approach
- **Hybrid approach**: Automated processing with confidence scoring
- **Manual review**: Low confidence cases flagged for human review
- **Feedback loop**: Errors used for training and workflow improvement

### Failure Handling
- **Structure detection failures**: Queue for manual review
- **Fallback processing**: Simple paragraph-based chunking as initial attempt
- **Error tracking**: Maintain logs for continuous improvement

### Acceptable Error Rates
- **TBD**: To be determined through initial implementation and testing
- **Iterative improvement**: Use feedback loop to reduce errors over time

## Query Strategy

### Retrieval Patterns
- **Hybrid approach**: Multiple chunk sizes for different query types
- **Multi-level search**: Search across different granularity levels
- **Context reconstruction**: Ability to show hierarchical context

### Query Classification Options
- **Phase 1**: Manual specification using prefixes (`/fact:`, `/concept:`, `/compare:`)
- **Phase 2**: Automatic detection (future enhancement)

### Query Routing Logic
```python
def route_query(query):
    if is_factual(query):
        return search_levels(["sentence", "paragraph"])
    elif is_conceptual(query):
        return search_levels(["paragraph", "section", "chapter"])
    elif is_comparative(query):
        return search_all_levels_with_context()
```

## Comment Thread Processing

### Thread Complexity
- **Nesting depth**: Up to 10 levels deep (e.g., Reddit posts)
- **Metadata preservation**: Timestamps, authors, parent-child relationships
- **Tree-aware chunking**: Preserve discussion context and hierarchy

## Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
- Build format-specific parsers
- Implement basic hierarchical chunking for org-roam
- Set up ChromaDB with metadata schema
- Test on small document subset
- Extract and index current reference patterns

**Deliverables:**
- Org-roam file parser with relationship detection
- Basic chunking at concept/relationship/content levels
- Reference extraction and indexing system
- ChromaDB schema implementation

### Phase 2: NLP Enhancement (Weeks 4-5)
- Add coreference resolution
- Implement confidence scoring for structure detection
- Build failure handling and manual review queue
- Integrate multi-format processing (PDF, HTML, EPUB)
- Create unified reference system

**Deliverables:**
- Coreference resolution pipeline
- Multi-format document processing
- Quality scoring and error handling
- Cross-document relationship mapping

### Phase 3: Multi-Model Embedding (Weeks 6-8)
- Test different embedding models for different chunk sizes
- Implement query routing logic
- Build context reconstruction for results
- Add comment thread processing
- Implement feedback loop system

**Deliverables:**
- Multi-level embedding system
- Query classification and routing
- Context reconstruction capabilities
- Comment thread processing pipeline

### Phase 4: Optimization and Enhancement (Future)
- Performance tuning for available hardware
- Automatic query classification
- Advanced relationship analysis
- Integration of RESEARCH todos
- Enhanced reference system automation

## Outstanding Questions and Future Considerations

### Technical Decisions Needed
1. **Query classification**: Start with manual or attempt automatic detection?
2. **Reference system priority**: How to weight different reference types in search results?
3. **Concept relationships**: Should search work across relationship types?
4. **TODO integration**: How to handle incomplete knowledge in RESEARCH todos?
5. **Backlink utilization**: Should backlinks influence relevance scoring?

### Future Enhancements
1. **Automated RESEARCH processing**: Auto-process URLs in RESEARCH todos
2. **Reference system evolution**: Migrate from "<23>" format to robust system
3. **Incremental processing**: Only reprocess changed documents
4. **Advanced relationship analysis**: Graph-based concept relationship discovery
5. **Integration with external sources**: Automated addition of new sources to zettelkasten

## Success Metrics

### Functional Metrics
- Successfully process all document formats
- Maintain hierarchical relationships in search results
- Preserve cross-document references
- Handle org-roam zettelkasten structure correctly

### Performance Metrics
- Process one book per day (target)
- Query response time under few seconds for live users
- Confidence scoring accuracy for structure detection

### Quality Metrics
- Reduction in manual review queue over time
- User satisfaction with search result relevance
- Successful context preservation in chunked results

## Risk Mitigation

### Technical Risks
- **Complex document structures**: Some formats may not follow expected patterns
- **Coreference resolution accuracy**: May introduce errors in context preservation
- **Scale performance**: Large document collections may impact query speed

### Mitigation Strategies
- **Fallback processing**: Simple chunking when structure detection fails
- **Confidence scoring**: Flag uncertain results for manual review
- **Incremental implementation**: Start with core functionality, add complexity gradually
- **Quality feedback loops**: Use errors to improve processing pipeline

## Hardware Utilization

### Available Resources
- Nvidia workstation GPUs (for embedding generation)
- Apple Silicon (for lighter NLP tasks)

### Optimization Strategy
- GPU acceleration for embedding generation
- Batch processing for efficiency
- Parallel processing for different document formats
- Memory-efficient processing for large document collections

## Related Documentation

### Conceptual Background
For detailed explanations of the concepts used in this project:

- **Chunking**: See [hierarchical-document-chunking-strategies.md](hierarchical-document-chunking-strategies.md)
- **Coreference Resolution**: See [coreference-resolution-nlp.md](coreference-resolution-nlp.md)
- **Embeddings**: See [semantic-search-embedding-strategies.md](semantic-search-embedding-strategies.md)
- **Query Routing**: See [query-classification-routing.md](query-classification-routing.md)
- **Org-roam Integration**: See [org-roam-zettelkasten-semantic-search.md](org-roam-zettelkasten-semantic-search.md)
- **ChromaDB Storage**: See [chromadb-hierarchical-storage.md](chromadb-hierarchical-storage.md)
- **NLP Tools**: See [nlp-pipeline-tools-overview.md](nlp-pipeline-tools-overview.md)
- **Multi-Format Processing**: See [multi-format-document-processing.md](multi-format-document-processing.md)

### Implementation Guides
*To be added as implementation phases are documented*
