# Phase 1: Foundation Implementation Guide
*Weeks 1-3: Core Zettelkasten Processing and Basic Infrastructure*

## Overview

Phase 1 establishes the foundational infrastructure for processing org-roam files, basic chunking, reference extraction, and ChromaDB integration. This phase focuses on getting the core zettelkasten functionality working before expanding to other formats.

## Technical Architecture

### Project Structure
```
semantic_processor/
├── src/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── org_mode.py
│   │   ├── base_parser.py
│   │   └── parser_factory.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── hierarchical_chunker.py
│   │   ├── zettelkasten_chunker.py
│   │   └── chunk_models.py
│   ├── references/
│   │   ├── __init__.py
│   │   ├── reference_extractor.py
│   │   ├── org_roam_links.py
│   │   └── bibliography.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── chromadb_client.py
│   │   └── metadata_schemas.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_utils.py
│   │   └── config.py
│   └── main.py
├── tests/
├── config/
│   └── settings.yaml
├── requirements.txt
└── README.md
```

## Core Components Implementation

### 1. Org-Mode Parser (`src/parsers/org_mode.py`)

```python
import orgparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

@dataclass
class OrgConcept:
    """Represents a single org-roam concept file"""
    file_path: Path
    title: str
    id: str
    content: str
    headlines: List[Dict[str, Any]]
    todo_keywords: List[str]
    org_roam_links: List[str]
    bibliography_refs: List[str]
    properties: Dict[str, str]
    backlinks: List[str]

@dataclass
class OrgHeadline:
    """Represents an org-mode headline with metadata"""
    level: int
    title: str
    content: str
    todo_state: Optional[str]
    properties: Dict[str, str]
    relationship_type: Optional[str]  # requirement, outcome, etc.

class OrgModeParser:
    """Parser for org-mode files with zettelkasten awareness"""

    def __init__(self, org_roam_directory: Path):
        self.org_roam_directory = org_roam_directory
        self.relationship_patterns = [
            r"^\*+ Requirement:",
            r"^\*+ Outcome:",
            r"^\*+ Definition:",
            r"^\*+ Example:",
            r"^\*+ Related:",
        ]

    def parse_file(self, file_path: Path) -> OrgConcept:
        """Parse a single org-roam file into structured data"""
        root = orgparse.load(file_path)

        concept = OrgConcept(
            file_path=file_path,
            title=self._extract_title(root),
            id=self._extract_id(root),
            content=self._extract_full_content(root),
            headlines=self._extract_headlines(root),
            todo_keywords=self._extract_todo_keywords(root),
            org_roam_links=self._extract_org_roam_links(root),
            bibliography_refs=self._extract_bibliography_refs(root),
            properties=self._extract_properties(root),
            backlinks=self._find_backlinks(file_path)
        )

        return concept

    def _extract_title(self, root) -> str:
        """Extract concept title from org file"""
        # Try TITLE property first
        if hasattr(root, 'properties') and 'TITLE' in root.properties:
            return root.properties['TITLE']

        # Try first headline
        for node in root:
            if hasattr(node, 'heading'):
                return node.heading

        # Fallback to filename
        return root.filename

    def _extract_id(self, root) -> str:
        """Extract org-roam ID"""
        if hasattr(root, 'properties') and 'ID' in root.properties:
            return root.properties['ID']
        return str(uuid.uuid4())

    def _extract_full_content(self, root) -> str:
        """Extract all text content from the file"""
        return root.get_body(format='plain')

    def _extract_todo_keywords(self, root) -> List[str]:
        """Extract all TODO keywords present in file"""
        keywords = set()
        for node in root[1:]:
            if hasattr(node, 'todo') and node.todo:
                keywords.add(node.todo)
        return list(keywords)

    def _extract_properties(self, root) -> Dict[str, str]:
        """Extract file-level properties"""
        if hasattr(root, 'properties'):
            return dict(root.properties)
        return {}

    def _extract_headlines(self, root) -> List[OrgHeadline]:
        """Extract all headlines with their content and metadata"""
        headlines = []

        def process_node(node, level=0):
            if hasattr(node, 'heading'):
                headline = OrgHeadline(
                    level=level,
                    title=node.heading,
                    content=self._get_node_content(node),
                    todo_state=getattr(node, 'todo', None),
                    properties=getattr(node, 'properties', {}),
                    relationship_type=self._detect_relationship_type(node.heading)
                )
                headlines.append(headline)

            # Process children
            for child in node:
                process_node(child, level + 1)

        process_node(root)
        return headlines

    def _get_node_content(self, node) -> str:
        """Extract text content from a node"""
        if hasattr(node, 'get_body'):
            return node.get_body(format='plain')
        return ""

    def _detect_relationship_type(self, heading: str) -> Optional[str]:
        """Detect if heading represents a relationship type"""
        heading_lower = heading.lower().strip()

        relationship_mapping = {
            'requirement': 'requirement',
            'requirements': 'requirement',
            'outcome': 'outcome',
            'outcomes': 'outcome',
            'definition': 'definition',
            'example': 'example',
            'examples': 'example',
            'related': 'related'
        }

        for pattern, rel_type in relationship_mapping.items():
            if pattern in heading_lower:
                return rel_type

        return None

    def _extract_org_roam_links(self, root) -> List[str]:
        """Extract org-roam style links [[file:...][...]]"""
        content = str(root)
        pattern = r'\[\[file:([^\]]+\.org)\](?:\[([^\]]*)\])?\]'
        matches = re.findall(pattern, content)
        return [match[0] for match in matches]

    def _extract_bibliography_refs(self, root) -> List[str]:
        """Extract bibliography references like <23>"""
        content = str(root)
        pattern = r'<(\d+)>'
        matches = re.findall(pattern, content)
        return [f"<{match}>" for match in matches]

    def _find_backlinks(self, file_path: Path) -> List[str]:
        """Find files that link to this file (simplified implementation)"""
        # This would need to scan the entire org-roam directory
        # For now, return empty list - implement in Phase 2
        return []
```

**Implementation Questions for Phase 1:**

1. **Custom TODO Keywords**: What are all your custom TODO states besides "RESEARCH" and "IN_PROGRESS"? Should they be weighted differently in search?

2. **Relationship Type Expansion**: Are there other relationship heading patterns beyond "Requirement:" and "Outcome:" that you use?

3. **Property Handling**: What org-mode properties do you use that should be preserved in metadata?

### 2. Zettelkasten Chunking Engine (`src/chunking/zettelkasten_chunker.py`)

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    id: str
    content: str
    level: str  # concept, relationship, content_block, sentence
    parent_id: Optional[str]
    child_ids: List[str]
    metadata: Dict[str, Any]
    confidence_score: float

class ZettelkastenChunker:
    """Hierarchical chunker specialized for zettelkasten structure"""

    def __init__(self, min_chunk_size: int = 50, max_chunk_size: int = 4000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_concept(self, concept: OrgConcept) -> List[Chunk]:
        """Create hierarchical chunks from an org-roam concept"""
        chunks = []

        # Level 1: Full concept chunk
        concept_chunk = self._create_concept_chunk(concept)
        chunks.append(concept_chunk)

        # Level 2: Relationship-based chunks
        relationship_chunks = self._create_relationship_chunks(
            concept, concept_chunk.id
        )
        chunks.extend(relationship_chunks)

        # Update parent's child IDs
        concept_chunk.child_ids = [c.id for c in relationship_chunks]

        # Level 3: Content block chunks
        for rel_chunk in relationship_chunks:
            content_chunks = self._create_content_block_chunks(
                rel_chunk.content, rel_chunk.id, concept
            )
            chunks.extend(content_chunks)
            rel_chunk.child_ids = [c.id for c in content_chunks]

        # Level 4: Sentence chunks (for precise retrieval)
        for chunk in chunks:
            if chunk.level == 'content_block':
                sentence_chunks = self._create_sentence_chunks(
                    chunk.content, chunk.id, concept
                )
                chunks.extend(sentence_chunks)
                chunk.child_ids = [c.id for c in sentence_chunks]

        return chunks

    def _create_concept_chunk(self, concept: OrgConcept) -> Chunk:
        """Create the top-level concept chunk"""
        # Include context about relationships and links
        context_info = self._build_concept_context(concept)

        full_content = f"{concept.title}\n\n{context_info}\n\n{concept.content}"

        return Chunk(
            id=str(uuid.uuid4()),
            content=full_content[:self.max_chunk_size],
            level='concept',
            parent_id=None,
            child_ids=[],
            metadata={
                'concept_title': concept.title,
                'concept_id': concept.id,
                'file_path': str(concept.file_path),
                'org_roam_links': concept.org_roam_links,
                'bibliography_refs': concept.bibliography_refs,
                'todo_keywords': concept.todo_keywords,
                'relationship_types': self._get_relationship_types(concept)
            },
            confidence_score=1.0  # Concept level always high confidence
        )

    def _create_relationship_chunks(self, concept: OrgConcept, parent_id: str) -> List[Chunk]:
        """Create chunks for each relationship type"""
        chunks = []
        relationship_content = {}

        # Group content by relationship type
        for headline in concept.headlines:
            if headline.relationship_type:
                rel_type = headline.relationship_type
                if rel_type not in relationship_content:
                    relationship_content[rel_type] = []
                relationship_content[rel_type].append(headline)

        # Create chunk for each relationship type
        for rel_type, headlines in relationship_content.items():
            content = self._combine_relationship_content(headlines, concept)

            if len(content.strip()) >= self.min_chunk_size:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=content,
                    level='relationship',
                    parent_id=parent_id,
                    child_ids=[],
                    metadata={
                        'concept_title': concept.title,
                        'concept_id': concept.id,
                        'relationship_type': rel_type,
                        'file_path': str(concept.file_path),
                        'headline_count': len(headlines)
                    },
                    confidence_score=self._calculate_relationship_confidence(headlines)
                )
                chunks.append(chunk)

        return chunks

    def _create_content_block_chunks(
        self,
        text: str,
        parent_id: str,
        concept: OrgConcept
    ) -> List[Chunk]:
        """Create paragraph-level chunks from text"""
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for para in paragraphs:
            if len(para) >= self.min_chunk_size:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=para,
                    level='content_block',
                    parent_id=parent_id,
                    child_ids=[],
                    metadata={
                        'concept_title': concept.title,
                        'concept_id': concept.id,
                        'file_path': str(concept.file_path),
                        'char_count': len(para)
                    },
                    confidence_score=0.9
                )
                chunks.append(chunk)

        return chunks

    def _create_sentence_chunks(
        self,
        text: str,
        parent_id: str,
        concept: OrgConcept
    ) -> List[Chunk]:
        """Create sentence-level chunks for precise retrieval"""
        chunks = []

        # Simple sentence splitting (improve in Phase 2 with spaCy)
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        for sentence in sentences:
            if len(sentence) >= self.min_chunk_size:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=sentence + '.',
                    level='sentence',
                    parent_id=parent_id,
                    child_ids=[],
                    metadata={
                        'concept_title': concept.title,
                        'concept_id': concept.id,
                        'file_path': str(concept.file_path)
                    },
                    confidence_score=0.85
                )
                chunks.append(chunk)

        return chunks

    def _build_concept_context(self, concept: OrgConcept) -> str:
        """Build contextual information about the concept"""
        context_parts = []

        if concept.org_roam_links:
            links_str = ', '.join(concept.org_roam_links[:5])  # Limit to first 5
            context_parts.append(f"Related concepts: {links_str}")

        if concept.bibliography_refs:
            refs_str = ', '.join(concept.bibliography_refs[:5])
            context_parts.append(f"References: {refs_str}")

        relationship_types = self._get_relationship_types(concept)
        if relationship_types:
            context_parts.append(f"Covers: {', '.join(relationship_types)}")

        return '\n'.join(context_parts)

    def _get_relationship_types(self, concept: OrgConcept) -> List[str]:
        """Extract unique relationship types from concept"""
        types = set()
        for headline in concept.headlines:
            if headline.relationship_type:
                types.add(headline.relationship_type)
        return list(types)

    def _combine_relationship_content(
        self,
        headlines: List[OrgHeadline],
        concept: OrgConcept
    ) -> str:
        """Combine content from multiple headlines of same relationship type"""
        combined = []
        for headline in headlines:
            if headline.content:
                combined.append(f"{headline.title}\n{headline.content}")
        return '\n\n'.join(combined)

    def _calculate_relationship_confidence(self, headlines: List[OrgHeadline]) -> float:
        """Calculate confidence score for relationship chunk"""
        # Higher confidence if:
        # - More headlines (more content)
        # - Headlines have TODO states (more structured)
        # - Headlines have properties

        base_confidence = 0.8

        # Bonus for multiple headlines
        if len(headlines) > 2:
            base_confidence += 0.05

        # Bonus for structured content
        has_todos = any(h.todo_state for h in headlines)
        if has_todos:
            base_confidence += 0.05

        has_properties = any(h.properties for h in headlines)
        if has_properties:
            base_confidence += 0.05

        return min(base_confidence, 1.0)
```

**Implementation Questions for Phase 1:**

4. **Chunk Size Optimization**: What's the optimal chunk size for your use case? Should different relationship types have different size limits?

5. **Confidence Scoring**: What factors should influence confidence scores? Presence of references, TODO states, content length?

6. **Context Inclusion**: How much context should be included in each chunk? Should related concept titles be included in content?

### 3. Reference System (`src/references/reference_extractor.py`)

```python
from typing import Dict, List, Set
from pathlib import Path
import re
from dataclasses import dataclass

@dataclass
class Reference:
    """Represents a reference with metadata"""
    ref_id: str
    ref_type: str  # org_roam_link, bibliography, url, backlink
    source_file: Path
    target: str
    context: str  # surrounding text
    confidence: float

class ReferenceExtractor:
    """Extracts and manages all types of references"""

    def __init__(self, org_roam_directory: Path):
        self.org_roam_directory = org_roam_directory
        self.reference_cache: Dict[str, List[Reference]] = {}

    def extract_all_references(self, concept: OrgConcept) -> Dict[str, List[Reference]]:
        """Extract all reference types from a concept"""
        references = {
            'org_roam_links': self._extract_org_roam_references(concept),
            'bibliography': self._extract_bibliography_references(concept),
            'urls': self._extract_url_references(concept),
            'backlinks': self._extract_backlinks(concept)
        }

        return references

    def _extract_org_roam_references(self, concept: OrgConcept) -> List[Reference]:
        """Extract org-roam links with context"""
        references = []

        for link in concept.org_roam_links:
            # Extract context around the link
            context = self._find_link_context(concept.content, link)

            references.append(Reference(
                ref_id=f"link_{hash(link)}",
                ref_type="org_roam_link",
                source_file=concept.file_path,
                target=link,
                context=context,
                confidence=1.0  # Explicit links have high confidence
            ))

        return references

    def _extract_bibliography_references(self, concept: OrgConcept) -> List[Reference]:
        """Extract and contextualize bibliography references"""
        references = []
        content = concept.content

        # Find all <number> patterns with context
        pattern = r'(.{0,50})<(\d+)>(.{0,50})'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            before_context = match.group(1).strip()
            ref_num = match.group(2)
            after_context = match.group(3).strip()

            full_context = f"{before_context} <{ref_num}> {after_context}".strip()

            references.append(Reference(
                ref_id=f"bib_{ref_num}",
                ref_type="bibliography",
                source_file=concept.file_path,
                target=f"<{ref_num}>",
                context=full_context,
                confidence=0.9  # High confidence for explicit references
            ))

        return references

    def _extract_url_references(self, concept: OrgConcept) -> List[Reference]:
        """Extract URLs, especially from RESEARCH TODOs"""
        references = []

        # URL pattern
        url_pattern = r'https?://[^\s\]<>]+'

        for headline in concept.headlines:
            if headline.todo_state == "RESEARCH":
                urls = re.findall(url_pattern, headline.content)
                for url in urls:
                    references.append(Reference(
                        ref_id=f"research_{hash(url)}",
                        ref_type="research_url",
                        source_file=concept.file_path,
                        target=url,
                        context=headline.content[:200],
                        confidence=0.7  # Medium confidence, needs processing
                    ))

        return references

    def _extract_backlinks(self, concept: OrgConcept) -> List[Reference]:
        """Extract backlinks (files linking to this concept)"""
        # Placeholder for Phase 2 - requires full directory scan
        return []

    def _find_link_context(self, content: str, link: str, context_size: int = 100) -> str:
        """Find surrounding context for a link"""
        # Find link in content
        pattern = re.escape(link)
        match = re.search(pattern, content)

        if match:
            start = max(0, match.start() - context_size)
            end = min(len(content), match.end() + context_size)
            return content[start:end].strip()

        return ""

    def build_reference_index(self, concepts: List[OrgConcept]) -> Dict[str, List[str]]:
        """Build index of all references across concepts"""
        index = {}

        for concept in concepts:
            refs = self.extract_all_references(concept)
            concept_id = concept.id

            # Index org-roam links
            for ref in refs['org_roam_links']:
                if ref.target not in index:
                    index[ref.target] = []
                index[ref.target].append(concept_id)

        return index

    def find_concept_neighbors(
        self,
        concept: OrgConcept,
        reference_index: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Find all concepts connected to this one"""
        neighbors = {
            'outgoing': concept.org_roam_links,
            'incoming': []
        }

        # Find incoming links (backlinks)
        concept_file = concept.file_path.name
        if concept_file in reference_index:
            neighbors['incoming'] = reference_index[concept_file]

        return neighbors
```

**Implementation Questions for Phase 1:**

7. **Bibliography Table Location**: Where is your bibliography org-mode table located? Should it be parsed separately to validate references?

8. **Reference Context Size**: How much context around references should be stored? Is 50 characters sufficient?

9. **Research URL Processing**: Should RESEARCH todos be processed immediately or queued for later batch processing?

### 4. ChromaDB Integration (`src/storage/chromadb_client.py`)

```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json

class ZettelkastenChromaDB:
    """ChromaDB client specialized for zettelkasten storage"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}
        self._initialize_collections()

    def _initialize_collections(self):
        """Initialize collections for different chunk levels"""
        collection_configs = {
            'concepts': 'Concept-level chunks',
            'relationships': 'Relationship-type chunks',
            'content_blocks': 'Paragraph-level chunks',
            'sentences': 'Sentence-level chunks'
        }

        for name, description in collection_configs.items():
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"description": description}
            )

    def store_concept_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Store chunks with their embeddings and metadata"""

        # Group chunks by level
        chunks_by_level = {}
        for chunk, embedding in zip(chunks, embeddings):
            level = chunk.level
            if level not in chunks_by_level:
                chunks_by_level[level] = {'chunks': [], 'embeddings': []}
            chunks_by_level[level]['chunks'].append(chunk)
            chunks_by_level[level]['embeddings'].append(embedding)

        # Store in appropriate collections
        for level, data in chunks_by_level.items():
            collection_name = self._get_collection_name(level)
            if collection_name in self.collections:
                self._store_in_collection(
                    collection_name,
                    data['chunks'],
                    data['embeddings']
                )

    def _get_collection_name(self, chunk_level: str) -> str:
        """Map chunk level to collection name"""
        level_map = {
            'concept': 'concepts',
            'relationship': 'relationships',
            'content_block': 'content_blocks',
            'sentence': 'sentences'
        }
        return level_map.get(chunk_level, 'content_blocks')

    def _store_in_collection(
        self,
        collection_name: str,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ):
        """Store chunks in specific collection"""
        collection = self.collections[collection_name]

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage"""
        metadata = chunk.metadata.copy()

        # Add standard fields
        metadata.update({
            'chunk_level': chunk.level,
            'parent_id': chunk.parent_id if chunk.parent_id else "",
            'confidence_score': chunk.confidence_score
        })

        # Handle child_ids separately (store count instead of full list)
        metadata['child_count'] = len(chunk.child_ids)

        # Convert lists to strings for ChromaDB compatibility
        for key, value in list(metadata.items()):
            if isinstance(value, list):
                metadata[key] = json.dumps(value)
            elif value is None:
                metadata[key] = ""
            elif isinstance(value, (int, float, str, bool)):
                pass  # These types are fine
            else:
                # Convert other types to string
                metadata[key] = str(value)

        return metadata

    def search_concepts(
        self,
        query_embedding: List[float],
        levels: List[str] = None,
        n_results: int = 10,
        where: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Search across specified chunk levels"""

        if levels is None:
            levels = ['concepts', 'relationships', 'content_blocks']

        results = {}
        for level in levels:
            if level in self.collections:
                collection_results = self.collections[level].query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where
                )
                results[level] = collection_results

        return results

    def get_chunk_with_hierarchy(self, chunk_id: str, collection_name: str) -> Dict[str, Any]:
        """Retrieve a chunk with its parent and children"""
        collection = self.collections.get(collection_name)
        if not collection:
            return None

        # Get the chunk
        result = collection.get(ids=[chunk_id])
        if not result or not result['ids']:
            return None

        chunk_data = {
            'id': result['ids'][0],
            'content': result['documents'][0],
            'metadata': result['metadatas'][0]
        }

        # Get parent if exists
        parent_id = result['metadatas'][0].get('parent_id')
        if parent_id:
            parent_collection = self._infer_parent_collection(collection_name)
            if parent_collection:
                parent_result = self.collections[parent_collection].get(ids=[parent_id])
                if parent_result and parent_result['ids']:
                    chunk_data['parent'] = parent_result

        return chunk_data

    def _infer_parent_collection(self, child_collection: str) -> Optional[str]:
        """Infer parent collection from child collection"""
        hierarchy = {
            'relationships': 'concepts',
            'content_blocks': 'relationships',
            'sentences': 'content_blocks'
        }
        return hierarchy.get(child_collection)

    def update_chunk(self, chunk_id: str, collection_name: str, new_metadata: Dict[str, Any]):
        """Update chunk metadata"""
        collection = self.collections.get(collection_name)
        if collection:
            collection.update(
                ids=[chunk_id],
                metadatas=[new_metadata]
            )

    def delete_concept(self, concept_id: str):
        """Delete all chunks related to a concept"""
        # Search for chunks with this concept_id and delete them
        for collection_name, collection in self.collections.items():
            # Query for chunks with this concept_id
            results = collection.get(
                where={"concept_id": concept_id}
            )
            if results and results['ids']:
                collection.delete(ids=results['ids'])
```

**Implementation Questions for Phase 1:**

10. **Embedding Model Selection**: Which specific sentence-transformers model should be used for Phase 1? Consider:
    - `all-MiniLM-L6-v2` (fast, 384 dimensions)
    - `all-MiniLM-L12-v2` (balanced, 384 dimensions)
    - `all-mpnet-base-v2` (best quality, 768 dimensions)

11. **Chunk Overlap Strategy**: Should chunks have overlapping content to preserve context at boundaries?

12. **Metadata Versioning**: How should you handle updates to org-roam files? Version chunks or replace entirely?

## Phase 1 Deliverables

### Week 1: Basic Infrastructure
- [ ] Project structure setup
- [ ] Org-mode parser implementation
- [ ] Basic chunk data models
- [ ] ChromaDB integration setup
- [ ] Configuration management

### Week 2: Core Processing Pipeline
- [ ] Zettelkasten-aware chunking
- [ ] Reference extraction system
- [ ] Metadata schema implementation
- [ ] Basic embedding integration
- [ ] Unit tests for core components

### Week 3: Integration and Testing
- [ ] End-to-end processing pipeline
- [ ] Test with sample org-roam files
- [ ] Error handling and logging
- [ ] Performance monitoring
- [ ] Documentation

## Testing Strategy

### Unit Tests
```python
import pytest
from pathlib import Path

def test_org_mode_parser():
    """Test parsing of sample org-roam file"""
    parser = OrgModeParser(Path("./test_data"))
    concept = parser.parse_file(Path("./test_data/sample.org"))

    assert concept.title is not None
    assert len(concept.headlines) > 0
    assert concept.id is not None

def test_zettelkasten_chunker():
    """Test hierarchical chunk creation"""
    chunker = ZettelkastenChunker()

    # Create mock concept
    concept = create_mock_concept()
    chunks = chunker.chunk_concept(concept)

    # Verify chunk levels exist
    levels = {chunk.level for chunk in chunks}
    assert 'concept' in levels
    assert 'relationship' in levels

    # Verify parent-child relationships
    concept_chunks = [c for c in chunks if c.level == 'concept']
    assert len(concept_chunks) == 1
    assert len(concept_chunks[0].child_ids) > 0

def test_reference_extractor():
    """Test reference extraction"""
    extractor = ReferenceExtractor(Path("./test_data"))
    concept = create_mock_concept()

    refs = extractor.extract_all_references(concept)

    assert 'org_roam_links' in refs
    assert 'bibliography' in refs
    assert 'urls' in refs
```

### Integration Tests
```python
def test_end_to_end_processing():
    """Test complete processing pipeline"""
    # Parse org file
    parser = OrgModeParser(Path("./test_data"))
    concept = parser.parse_file(Path("./test_data/sample.org"))

    # Chunk concept
    chunker = ZettelkastenChunker()
    chunks = chunker.chunk_concept(concept)

    # Extract references
    extractor = ReferenceExtractor(Path("./test_data"))
    refs = extractor.extract_all_references(concept)

    # Store in ChromaDB (mock embeddings)
    db = ZettelkastenChromaDB("./test_chroma_db")
    mock_embeddings = [[0.1] * 384 for _ in chunks]
    db.store_concept_chunks(chunks, mock_embeddings)

    # Verify storage
    assert len(chunks) > 0
```

## Configuration Management

### Sample `config/settings.yaml`
```yaml
org_roam:
  directory: "/path/to/org-roam"
  file_extensions: [".org"]

chunking:
  min_chunk_size: 50
  max_chunk_size: 4000
  overlap_percentage: 0.1

embedding:
  model_name: "all-MiniLM-L12-v2"
  batch_size: 32
  device: "cuda"  # or "cpu"

chromadb:
  persist_directory: "./chroma_db"
  collection_prefix: "zettelkasten"

processing:
  confidence_threshold: 0.7
  enable_coreference: false  # Phase 2 feature

logging:
  level: "INFO"
  file: "./logs/processor.log"
```

### Configuration Loading (`src/utils/config.py`)
```python
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """Application configuration"""
    org_roam_directory: Path
    min_chunk_size: int
    max_chunk_size: int
    embedding_model: str
    chromadb_directory: str
    confidence_threshold: float

def load_config(config_path: str = "config/settings.yaml") -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return Config(
        org_roam_directory=Path(data['org_roam']['directory']),
        min_chunk_size=data['chunking']['min_chunk_size'],
        max_chunk_size=data['chunking']['max_chunk_size'],
        embedding_model=data['embedding']['model_name'],
        chromadb_directory=data['chromadb']['persist_directory'],
        confidence_threshold=data['processing']['confidence_threshold']
    )
```

## Future Capabilities (Post Phase 1)

### Advanced Reference Resolution
- **Automatic Bibliography Lookup**: Resolve "<23>" references to actual citation data
- **Dead Link Detection**: Identify broken org-roam links
- **Reference Validation**: Verify bibliography entries exist

### Enhanced Chunking
- **Semantic Boundary Detection**: Use NLP to find natural breaking points
- **Content-Aware Sizing**: Adjust chunk sizes based on content density
- **Cross-Reference Preservation**: Maintain reference context across chunks

### Query Enhancement
- **Graph-Aware Search**: Leverage org-roam link structure in search
- **Temporal Filtering**: Search based on file modification dates
- **TODO State Filtering**: Filter results by completion status

### Performance Optimization
- **Incremental Processing**: Only process changed files
- **Parallel Processing**: Multi-threaded chunk processing
- **Caching Strategy**: Cache embeddings and parsed structures

## Related Documentation

For conceptual background on the techniques used in this implementation:

- **Chunking Concepts**: [hierarchical-document-chunking-strategies.md](hierarchical-document-chunking-strategies.md)
- **Org-roam Specifics**: [org-roam-zettelkasten-semantic-search.md](org-roam-zettelkasten-semantic-search.md)
- **ChromaDB Usage**: [chromadb-hierarchical-storage.md](chromadb-hierarchical-storage.md)
- **Embeddings**: [semantic-search-embedding-strategies.md](semantic-search-embedding-strategies.md)
- **NLP Tools**: [nlp-pipeline-tools-overview.md](nlp-pipeline-tools-overview.md)

For project overview and full implementation roadmap:

- **Project Specification**: [semantic-document-processing-project-specification.md](semantic-document-processing-project-specification.md)
