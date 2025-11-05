# Phase 2: NLP Enhancement & Multi-Format Implementation Guide
*Weeks 4-5: Coreference Resolution, Multi-Format Support, and Quality Control*

## Overview

Phase 2 expands the system to handle multiple document formats, implements advanced NLP processing including coreference resolution, and establishes quality control mechanisms. This phase bridges the gap between basic zettelkasten processing and a comprehensive multi-format semantic search system.

## Technical Architecture Expansion

### Enhanced Project Structure
```
semantic_processor/
├── src/
│   ├── parsers/
│   │   ├── pdf_parser.py          # NEW
│   │   ├── html_parser.py         # NEW
│   │   ├── epub_parser.py         # NEW
│   │   ├── markdown_parser.py     # NEW
│   │   └── comment_thread_parser.py # NEW
│   ├── nlp/                       # NEW
│   │   ├── __init__.py
│   │   ├── coreference_resolver.py
│   │   ├── sentence_processor.py
│   │   └── context_enhancer.py
│   ├── quality/                   # NEW
│   │   ├── __init__.py
│   │   ├── confidence_scorer.py
│   │   ├── structure_validator.py
│   │   └── manual_review_queue.py
│   ├── chunking/
│   │   ├── multi_format_chunker.py # NEW
│   │   └── adaptive_chunker.py     # NEW
│   └── references/
│       ├── cross_doc_resolver.py   # NEW
│       └── reference_graph.py      # NEW
```

## Core NLP Components

### 1. Coreference Resolution System (`src/nlp/coreference_resolver.py`)

```python
import spacy
from spacy.tokens import Doc, Span
from typing import List, Dict, Tuple, Optional
import neuralcoref  # or alternative coreference model
from dataclasses import dataclass

@dataclass
class CoreferenceResolution:
    """Result of coreference resolution with confidence score"""
    original_text: str
    resolved_text: str
    replacements: List[Tuple[str, str, float]]  # (original, replacement, confidence)
    confidence_score: float

class CoreferenceResolver:
    """Advanced coreference resolution for context preservation"""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

        # Add coreference resolution component
        try:
            neuralcoref.add_to_pipe(self.nlp)
        except Exception as e:
            print(f"Warning: Could not load neuralcoref: {e}")
            self.coreference_available = False
        else:
            self.coreference_available = True

    def resolve_coreferences(
        self,
        text: str,
        context: Optional[str] = None
    ) -> CoreferenceResolution:
        """Resolve coreferences in text with optional context"""

        if not self.coreference_available:
            return CoreferenceResolution(
                original_text=text,
                resolved_text=text,
                replacements=[],
                confidence_score=1.0
            )

        # Combine context with text for better resolution
        full_text = f"{context}\n\n{text}" if context else text
        context_offset = len(context) + 2 if context else 0

        doc = self.nlp(full_text)

        if not doc._.has_coref:
            return CoreferenceResolution(
                original_text=text,
                resolved_text=text,
                replacements=[],
                confidence_score=1.0
            )

        # Extract coreference clusters
        replacements = []
        resolved_text = text

        for cluster in doc._.coref_clusters:
            main_mention = cluster.main

            # Find mentions that need resolution in our target text
            for mention in cluster.mentions:
                mention_start = mention.start_char - context_offset
                mention_end = mention.end_char - context_offset

                # Skip if mention is outside our target text
                if mention_start < 0 or mention_end > len(text):
                    continue

                # Skip if mention is the main reference
                if mention.text == main_mention.text:
                    continue

                # Calculate replacement confidence
                confidence = self._calculate_replacement_confidence(
                    mention, main_mention, doc
                )

                # Perform replacement if confidence is high enough
                if confidence > 0.7:  # Configurable threshold
                    original_mention = mention.text
                    replacement = main_mention.text

                    resolved_text = resolved_text.replace(
                        original_mention, replacement, 1
                    )

                    replacements.append((
                        original_mention,
                        replacement,
                        confidence
                    ))

        overall_confidence = self._calculate_overall_confidence(replacements)

        return CoreferenceResolution(
            original_text=text,
            resolved_text=resolved_text,
            replacements=replacements,
            confidence_score=overall_confidence
        )

    def _calculate_replacement_confidence(
        self,
        mention: Span,
        main_mention: Span,
        doc: Doc
    ) -> float:
        """Calculate confidence score for a coreference replacement"""
        confidence = 0.8  # Base confidence

        # Boost confidence for clear pronoun replacements
        if mention.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them']:
            confidence += 0.1

        # Reduce confidence for ambiguous cases
        if mention.text.lower() in ['this', 'that', 'these', 'those']:
            confidence -= 0.2

        # Consider distance between mentions
        distance = abs(mention.start - main_mention.start)
        if distance > 100:  # tokens
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _calculate_overall_confidence(self, replacements: List[Tuple[str, str, float]]) -> float:
        """Calculate overall confidence from individual replacements"""
        if not replacements:
            return 1.0

        confidences = [conf for _, _, conf in replacements]
        return sum(confidences) / len(confidences)

    def resolve_chunk_with_context(
        self,
        chunk_text: str,
        parent_context: str,
        concept_context: str
    ) -> CoreferenceResolution:
        """Resolve coreferences using hierarchical context"""

        # Build context hierarchy: concept → parent → chunk
        full_context = f"{concept_context}\n\n{parent_context}"

        return self.resolve_coreferences(chunk_text, full_context)
```

**Implementation Questions for Phase 2:**

13. **Coreference Resolution Scope**: Should coreference resolution be applied to all chunk levels or only to smaller chunks (paragraphs/sentences)?

14. **Context Window Size**: How much parent context should be used for coreference resolution? Full parent chunk or limited window?

15. **Confidence Thresholds**: What confidence threshold should trigger manual review? Should different document types have different thresholds?

### 2. Multi-Format Parser Factory (`src/parsers/parser_factory.py`)

```python
from pathlib import Path
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import mimetypes

from .org_mode import OrgModeParser, OrgConcept
from .pdf_parser import PDFParser, PDFDocument
from .html_parser import HTMLParser, HTMLDocument
from .epub_parser import EPUBParser, EPUBDocument
from .markdown_parser import MarkdownParser, MarkdownDocument
from .comment_thread_parser import CommentThreadParser, CommentThread

class ParsedDocument(ABC):
    """Base class for all parsed document types"""

    @abstractmethod
    def get_hierarchical_structure(self) -> Dict[str, Any]:
        """Return hierarchical structure for chunking"""
        pass

    @abstractmethod
    def get_content(self) -> str:
        """Return full document content"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return document metadata"""
        pass

class DocumentParserFactory:
    """Factory for creating appropriate parsers based on file type"""

    def __init__(self, org_roam_directory: Optional[Path] = None):
        self.org_roam_directory = org_roam_directory
        self.parsers = {}
        self._initialize_parsers()

    def _initialize_parsers(self):
        """Initialize all available parsers"""
        self.parsers = {
            '.org': OrgModeParser(self.org_roam_directory) if self.org_roam_directory else None,
            '.md': MarkdownParser(),
            '.markdown': MarkdownParser(),
            '.pdf': PDFParser(),
            '.html': HTMLParser(),
            '.htm': HTMLParser(),
            '.epub': EPUBParser(),
            'comment_thread': CommentThreadParser()  # Special case for JSON comment data
        }

    def parse_document(
        self,
        file_path: Union[Path, str],
        document_type: Optional[str] = None
    ) -> ParsedDocument:
        """Parse document using appropriate parser"""

        file_path = Path(file_path)

        # Determine document type
        if document_type:
            parser_key = document_type
        else:
            parser_key = file_path.suffix.lower()

        # Special handling for comment threads (JSON files)
        if parser_key not in self.parsers:
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(str(file_path))
            parser_key = self._mime_to_parser_key(mime_type)

        if parser_key not in self.parsers or self.parsers[parser_key] is None:
            raise ValueError(f"No parser available for file type: {parser_key}")

        parser = self.parsers[parser_key]
        return parser.parse_file(file_path)

    def _mime_to_parser_key(self, mime_type: Optional[str]) -> str:
        """Convert MIME type to parser key"""
        mime_mapping = {
            'application/pdf': '.pdf',
            'text/html': '.html',
            'application/epub+zip': '.epub',
            'text/markdown': '.md',
            'text/plain': '.md'  # Fallback for plain text
        }
        return mime_mapping.get(mime_type, 'unknown')
```

### 3. PDF Parser with Structure Detection (`src/parsers/pdf_parser.py`)

```python
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import uuid

@dataclass
class PDFSection:
    """Represents a section in a PDF document"""
    title: str
    level: int
    page_start: int
    page_end: int
    content: str
    subsections: List['PDFSection']

@dataclass
class PDFDocument:
    """Represents a parsed PDF document"""
    file_path: Path
    title: str
    authors: List[str]
    abstract: str
    sections: List[PDFSection]
    references: List[str]
    figures: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    full_text: str
    metadata: Dict[str, Any]

    def get_hierarchical_structure(self) -> Dict[str, Any]:
        return {
            'type': 'pdf_document',
            'title': self.title,
            'sections': [self._section_to_dict(s) for s in self.sections],
            'has_abstract': bool(self.abstract),
            'has_references': bool(self.references)
        }

    def get_content(self) -> str:
        return self.full_text

    def get_metadata(self) -> Dict[str, Any]:
        return {
            **self.metadata,
            'authors': self.authors,
            'section_count': len(self.sections),
            'page_count': self.metadata.get('page_count', 0)
        }

    def _section_to_dict(self, section: PDFSection) -> Dict[str, Any]:
        return {
            'title': section.title,
            'level': section.level,
            'page_start': section.page_start,
            'page_end': section.page_end,
            'subsection_count': len(section.subsections),
            'subsections': [self._section_to_dict(s) for s in section.subsections]
        }

class PDFParser:
    """Advanced PDF parser with structure detection"""

    def __init__(self):
        self.section_patterns = [
            r'^(\d+\.?\d*\.?\d*)\s+([A-Z][^.\n]{10,100})',  # Numbered sections
            r'^([A-Z][A-Z\s]{5,50})',  # All caps headers
            r'^(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References)',  # Standard paper sections
        ]
        self.confidence_threshold = 0.6

    def parse_file(self, file_path: Path) -> PDFDocument:
        """Parse PDF file with structure detection"""

        # Extract text and metadata
        full_text, metadata = self._extract_text_and_metadata(file_path)

        # Detect document structure
        sections = self._detect_sections(full_text)

        # Extract special elements
        abstract = self._extract_abstract(full_text)
        references = self._extract_references(full_text)
        title, authors = self._extract_title_and_authors(full_text, metadata)

        return PDFDocument(
            file_path=file_path,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            references=references,
            figures=[],  # TODO: Implement figure extraction
            tables=[],   # TODO: Implement table extraction
            full_text=full_text,
            metadata=metadata
        )

    def _extract_text_and_metadata(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF"""
        full_text = ""
        metadata = {}

        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = {
                    'page_count': len(pdf.pages),
                    'creator': pdf.metadata.get('Creator', ''),
                    'producer': pdf.metadata.get('Producer', ''),
                    'creation_date': pdf.metadata.get('CreationDate', ''),
                }

                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\n[Page {page_num + 1}]\n{page_text}"

        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            # Fallback to PyPDF2
            full_text, fallback_metadata = self._extract_with_pypdf2(file_path)
            metadata.update(fallback_metadata)

        return full_text, metadata

    def _extract_with_pypdf2(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Fallback extraction using PyPDF2"""
        full_text = ""
        metadata = {}

        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                metadata = {
                    'page_count': len(reader.pages),
                    'fallback_parser': 'PyPDF2'
                }

                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\n[Page {page_num + 1}]\n{page_text}"
        except Exception as e:
            print(f"PyPDF2 fallback failed: {e}")

        return full_text, metadata

    def _detect_sections(self, text: str) -> List[PDFSection]:
        """Detect document sections using pattern matching"""
        sections = []
        lines = text.split('\n')
        current_section = None
        section_confidence_scores = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if line matches section pattern
            section_match, confidence = self._is_section_header(line, i, lines)

            if section_match and confidence > self.confidence_threshold:
                # Save previous section
                if current_section:
                    current_section.content = current_section.content.strip()
                    sections.append(current_section)

                # Start new section
                current_section = PDFSection(
                    title=line,
                    level=self._determine_section_level(line),
                    page_start=self._estimate_page_number(i, lines),
                    page_end=0,  # Will be set when section ends
                    content="",
                    subsections=[]
                )
                section_confidence_scores.append(confidence)

            elif current_section:
                # Add content to current section
                current_section.content += line + "\n"

        # Close final section
        if current_section:
            current_section.content = current_section.content.strip()
            sections.append(current_section)

        # Set end pages for sections
        for i, section in enumerate(sections[:-1]):
            section.page_end = sections[i + 1].page_start - 1
        if sections:
            sections[-1].page_end = self._estimate_total_pages(lines)

        # Calculate overall structure confidence
        avg_confidence = sum(section_confidence_scores) / len(section_confidence_scores) if section_confidence_scores else 0

        return sections if avg_confidence > self.confidence_threshold else []

    def _is_section_header(self, line: str, line_index: int, all_lines: List[str]) -> Tuple[bool, float]:
        """Determine if line is a section header with confidence score"""
        confidence = 0.0

        # Check against patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                confidence += 0.4

        # Heuristic checks
        if len(line) < 100:  # Headers are usually short
            confidence += 0.1

        if line.isupper() and len(line.split()) > 1:  # All caps multi-word
            confidence += 0.2

        if line.endswith('.') and not line.endswith('..'):  # Doesn't end with period
            confidence -= 0.2

        # Check context - next line should be content or empty
        if line_index + 1 < len(all_lines):
            next_line = all_lines[line_index + 1].strip()
            if next_line and not next_line.isupper():
                confidence += 0.1

        return confidence > 0.3, confidence

    def _determine_section_level(self, title: str) -> int:
        """Determine hierarchical level of section"""
        # Check for numbered sections (e.g., "1.2.3 Section Title")
        numbering_match = re.match(r'^(\d+\.?\d*\.?\d*)', title)
        if numbering_match:
            numbering = numbering_match.group(1)
            level = numbering.count('.') + 1
            return level

        # Default level based on formatting
        if title.isupper():
            return 1
        return 2

    def _estimate_page_number(self, line_index: int, all_lines: List[str]) -> int:
        """Estimate page number from line index"""
        # Look backward for page markers
        for i in range(line_index, max(0, line_index - 100), -1):
            line = all_lines[i]
            page_match = re.search(r'\[Page (\d+)\]', line)
            if page_match:
                return int(page_match.group(1))
        return 1

    def _estimate_total_pages(self, all_lines: List[str]) -> int:
        """Estimate total number of pages"""
        max_page = 1
        for line in all_lines:
            page_match = re.search(r'\[Page (\d+)\]', line)
            if page_match:
                max_page = max(max_page, int(page_match.group(1)))
        return max_page

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from document text"""
        # Look for abstract section
        abstract_pattern = r'(?i)abstract\s*\n(.*?)(?=\n\s*(?:introduction|keywords|1\.|\n\n[A-Z]))'
        match = re.search(abstract_pattern, text, re.DOTALL)

        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\n+', ' ', abstract)
            return abstract

        return ""

    def _extract_references(self, text: str) -> List[str]:
        """Extract references from document"""
        references = []

        # Look for references section
        ref_pattern = r'(?i)references?\s*\n(.*?)(?=\Z|\n\s*(?:appendix|index))'
        match = re.search(ref_pattern, text, re.DOTALL)

        if match:
            ref_text = match.group(1)
            # Split by common reference patterns
            ref_lines = re.split(r'\n(?=\[\d+\]|\d+\.)', ref_text)

            for ref in ref_lines:
                ref = ref.strip()
                if len(ref) > 20:  # Filter out short non-references
                    references.append(ref)

        return references[:50]  # Limit to first 50 references

    def _extract_title_and_authors(self, text: str, metadata: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Extract title and authors from document"""
        title = metadata.get('Title', '')
        authors = []

        # Try to extract from first few lines if not in metadata
        if not title:
            lines = text.split('\n')
            for i, line in enumerate(lines[:10]):
                line = line.strip()
                if len(line) > 10 and not line.startswith('[Page'):
                    title = line
                    break

        return title, authors
```

**Implementation Questions for Phase 2:**

16. **PDF Structure Detection**: What confidence threshold should trigger fallback to simple paragraph chunking for PDFs with poor structure detection?

17. **Academic Paper vs Manual Distinction**: Should different PDF types (academic papers vs manuals) use different parsing strategies?

18. **Figure and Table Handling**: Should figures and tables be processed as separate searchable entities or embedded in text chunks?

### 4. Quality Control System (`src/quality/confidence_scorer.py`)

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import statistics
import json
import time
from datetime import datetime
from pathlib import Path

class QualityIssue(Enum):
    LOW_STRUCTURE_CONFIDENCE = "low_structure_confidence"
    COREFERENCE_UNCERTAINTY = "coreference_uncertainty"
    MISSING_REFERENCES = "missing_references"
    PARSING_ERRORS = "parsing_errors"
    CONTENT_TOO_SHORT = "content_too_short"
    ENCODING_ISSUES = "encoding_issues"

@dataclass
class QualityAssessment:
    """Assessment of document processing quality"""
    overall_confidence: float
    chunk_confidences: List[float]
    issues: List[QualityIssue]
    manual_review_recommended: bool
    processing_notes: List[str]

class ConfidenceScorer:
    """Calculates confidence scores for processed documents"""

    def __init__(self):
        self.thresholds = {
            'manual_review': 0.7,
            'structure_detection': 0.6,
            'coreference_resolution': 0.75,
            'min_content_length': 50
        }

    def assess_document_quality(
        self,
        parsed_doc,
        chunks: List,
        coreference_results: Optional[List] = None
    ) -> QualityAssessment:
        """Comprehensive quality assessment of processed document"""

        issues = []
        processing_notes = []
        confidence_scores = []

        # Structure detection confidence
        structure_confidence = self._assess_structure_quality(parsed_doc)
        confidence_scores.append(structure_confidence)

        if structure_confidence < self.thresholds['structure_detection']:
            issues.append(QualityIssue.LOW_STRUCTURE_CONFIDENCE)
            processing_notes.append(f"Structure detection confidence: {structure_confidence:.2f}")

        # Chunk quality assessment
        chunk_confidences = []
        for chunk in chunks:
            chunk_conf = self._assess_chunk_quality(chunk)
            chunk_confidences.append(chunk_conf)
            confidence_scores.append(chunk_conf)

            if len(chunk.content) < self.thresholds['min_content_length']:
                issues.append(QualityIssue.CONTENT_TOO_SHORT)

        # Coreference resolution assessment
        if coreference_results:
            coref_confidence = self._assess_coreference_quality(coreference_results)
            confidence_scores.append(coref_confidence)

            if coref_confidence < self.thresholds['coreference_resolution']:
                issues.append(QualityIssue.COREFERENCE_UNCERTAINTY)
                processing_notes.append(f"Coreference resolution confidence: {coref_confidence:.2f}")

        # Overall confidence calculation
        overall_confidence = statistics.harmonic_mean(confidence_scores) if confidence_scores else 0.0

        # Manual review recommendation
        manual_review = (
            overall_confidence < self.thresholds['manual_review'] or
            QualityIssue.LOW_STRUCTURE_CONFIDENCE in issues or
            len(issues) >= 3
        )

        return QualityAssessment(
            overall_confidence=overall_confidence,
            chunk_confidences=chunk_confidences,
            issues=issues,
            manual_review_recommended=manual_review,
            processing_notes=processing_notes
        )

    def _assess_structure_quality(self, parsed_doc) -> float:
        """Assess quality of document structure detection"""
        structure = parsed_doc.get_hierarchical_structure()
        confidence = 0.5  # Base confidence

        # Boost for detected sections
        if 'sections' in structure:
            section_count = len(structure['sections'])
            if section_count > 0:
                confidence += min(0.3, section_count * 0.05)

        # Boost for metadata richness
        metadata = parsed_doc.get_metadata()
        if metadata.get('title'):
            confidence += 0.1
        if metadata.get('authors'):
            confidence += 0.05

        # Document type specific adjustments
        doc_type = structure.get('type', '')
        if doc_type == 'org-roam':
            confidence += 0.2  # Org-roam has reliable structure
        elif doc_type == 'pdf_document':
            # PDF confidence depends on section detection
            if structure.get('sections'):
                confidence += 0.1
            else:
                confidence -= 0.2

        return min(1.0, confidence)

    def _assess_chunk_quality(self, chunk) -> float:
        """Assess quality of individual chunk"""
        confidence = chunk.confidence_score

        # Adjust based on content characteristics
        content_length = len(chunk.content)
        if content_length < 20:
            confidence *= 0.5
        elif content_length > 5000:
            confidence *= 0.8  # Very long chunks may lose coherence

        # Adjust based on metadata completeness
        if chunk.metadata.get('hierarchy_path'):
            confidence += 0.05

        if chunk.parent_id or chunk.child_ids:
            confidence += 0.05  # Has structural relationships

        return min(1.0, confidence)

    def _assess_coreference_quality(self, coreference_results: List) -> float:
        """Assess quality of coreference resolution"""
        if not coreference_results:
            return 1.0  # No coreference needed

        confidences = [cr.confidence_score for cr in coreference_results]
        return statistics.mean(confidences) if confidences else 1.0

class ManualReviewQueue:
    """Manages documents flagged for manual review"""

    def __init__(self, queue_directory: Path):
        self.queue_directory = Path(queue_directory)
        self.queue_directory.mkdir(exist_ok=True)

    def add_to_queue(
        self,
        document_path: Path,
        quality_assessment: QualityAssessment,
        chunks: List
    ):
        """Add document to manual review queue"""

        review_item = {
            'original_document': str(document_path),
            'timestamp': datetime.now().isoformat(),
            'quality_assessment': {
                'confidence': quality_assessment.overall_confidence,
                'issues': [issue.value for issue in quality_assessment.issues],
                'notes': quality_assessment.processing_notes
            },
            'chunk_count': len(chunks),
            'chunk_confidences': quality_assessment.chunk_confidences,
            'review_status': 'pending'
        }

        # Save review item
        review_file = self.queue_directory / f"review_{document_path.stem}_{int(time.time())}.json"
        with open(review_file, 'w') as f:
            json.dump(review_item, f, indent=2)

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all pending manual reviews"""
        pending = []

        for review_file in self.queue_directory.glob("review_*.json"):
            with open(review_file, 'r') as f:
                review_item = json.load(f)
                if review_item.get('review_status') == 'pending':
                    pending.append({
                        **review_item,
                        'review_file': review_file
                    })

        return sorted(pending, key=lambda x: x['timestamp'])

    def mark_reviewed(self, review_file: Path, notes: str = ""):
        """Mark a document as manually reviewed"""
        with open(review_file, 'r') as f:
            review_item = json.load(f)

        review_item['review_status'] = 'completed'
        review_item['review_date'] = datetime.now().isoformat()
        review_item['review_notes'] = notes

        with open(review_file, 'w') as f:
            json.dump(review_item, f, indent=2)
```

### 5. Enhanced Multi-Format Chunking (`src/chunking/multi_format_chunker.py`)

```python
from typing import List, Dict, Any, Optional, Type
from abc import ABC, abstractmethod
import uuid

class FormatSpecificChunker(ABC):
    """Base class for format-specific chunking strategies"""

    @abstractmethod
    def chunk_document(self, document) -> List:
        """Chunk document according to format-specific strategy"""
        pass

class PDFChunker(FormatSpecificChunker):
    """Chunker specialized for PDF documents"""

    def __init__(self):
        self.target_chunk_sizes = {
            'document': (3000, 5000),
            'section': (1000, 2000),
            'paragraph': (200, 800),
            'sentence': (20, 150)
        }

    def chunk_document(self, document) -> List:
        """Chunk PDF using section-aware strategy"""
        chunks = []

        # Document-level chunk
        doc_chunk = self._create_document_chunk(document)
        chunks.append(doc_chunk)

        # Section-level chunks
        for section in document.sections:
            section_chunks = self._chunk_section(section, document, doc_chunk.id)
            chunks.extend(section_chunks)

        # Abstract as special chunk if present
        if document.abstract:
            abstract_chunk = self._create_abstract_chunk(document, doc_chunk.id)
            chunks.append(abstract_chunk)

        return chunks

    def _create_document_chunk(self, document):
        """Create document-level overview chunk"""
        from .chunk_models import Chunk

        # Combine title, abstract, and section summaries
        content_parts = [document.title]

        if document.abstract:
            content_parts.append(f"Abstract: {document.abstract}")

        # Add section overview
        section_titles = [s.title for s in document.sections]
        if section_titles:
            content_parts.append(f"Sections: {', '.join(section_titles)}")

        content = '\n\n'.join(content_parts)

        return Chunk(
            id=str(uuid.uuid4()),
            content=content[:self.target_chunk_sizes['document'][1]],
            level='document',
            parent_id=None,
            child_ids=[],
            metadata={
                'document_type': 'pdf',
                'title': document.title,
                'authors': document.authors,
                'section_count': len(document.sections),
                'has_abstract': bool(document.abstract),
                'file_path': str(document.file_path)
            },
            confidence_score=0.9
        )

    def _create_abstract_chunk(self, document, parent_id: str):
        """Create special chunk for abstract"""
        from .chunk_models import Chunk

        return Chunk(
            id=str(uuid.uuid4()),
            content=document.abstract,
            level='abstract',
            parent_id=parent_id,
            child_ids=[],
            metadata={
                'document_type': 'pdf',
                'title': document.title,
                'special_section': 'abstract',
                'file_path': str(document.file_path)
            },
            confidence_score=0.95
        )

    def _chunk_section(self, section, document, parent_id: str) -> List:
        """Chunk a PDF section into paragraphs"""
        from .chunk_models import Chunk
        chunks = []

        # Section-level chunk
        section_chunk = Chunk(
            id=str(uuid.uuid4()),
            content=f"{section.title}\n\n{section.content[:self.target_chunk_sizes['section'][1]]}",
            level='section',
            parent_id=parent_id,
            child_ids=[],
            metadata={
                'document_type': 'pdf',
                'document_title': document.title,
                'section_title': section.title,
                'section_level': section.level,
                'pages': f"{section.page_start}-{section.page_end}",
                'file_path': str(document.file_path)
            },
            confidence_score=0.85
        )
        chunks.append(section_chunk)

        # Paragraph chunks from section content
        paragraphs = [p.strip() for p in section.content.split('\n\n') if p.strip()]
        for para in paragraphs:
            if len(para) >= 50:
                para_chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=para,
                    level='paragraph',
                    parent_id=section_chunk.id,
                    child_ids=[],
                    metadata={
                        'document_type': 'pdf',
                        'document_title': document.title,
                        'section_title': section.title,
                        'file_path': str(document.file_path)
                    },
                    confidence_score=0.8
                )
                chunks.append(para_chunk)
                section_chunk.child_ids.append(para_chunk.id)

        return chunks

class MultiFormatChunker:
    """Main chunker that delegates to format-specific chunkers"""

    def __init__(self):
        from .zettelkasten_chunker import ZettelkastenChunker

        self.chunkers: Dict[str, FormatSpecificChunker] = {
            'org-roam': ZettelkastenChunker(),
            'pdf_document': PDFChunker(),
            # Add other format chunkers as implemented
        }

    def chunk_document(
        self,
        document,
        apply_coreference: bool = True
    ) -> List:
        """Chunk document using appropriate strategy"""

        doc_type = document.get_hierarchical_structure().get('type', 'unknown')

        if doc_type not in self.chunkers:
            # Fallback to generic chunking
            return self._generic_chunk(document)

        # Use specialized chunker
        chunker = self.chunkers[doc_type]
        chunks = chunker.chunk_document(document)

        # Apply coreference resolution if requested
        if apply_coreference:
            chunks = self._apply_coreference_resolution(chunks, document)

        return chunks

    def _generic_chunk(self, document) -> List:
        """Generic chunking fallback for unknown formats"""
        from .chunk_models import Chunk

        content = document.get_content()
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        chunks = []
        for para in paragraphs:
            if len(para) >= 50:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=para,
                    level='paragraph',
                    parent_id=None,
                    child_ids=[],
                    metadata=document.get_metadata(),
                    confidence_score=0.6
                )
                chunks.append(chunk)

        return chunks

    def _apply_coreference_resolution(self, chunks: List, document) -> List:
        """Apply coreference resolution to chunks"""
        from ..nlp.coreference_resolver import CoreferenceResolver

        resolver = CoreferenceResolver()
        resolved_chunks = []

        # Build context hierarchy for better resolution
        chunk_hierarchy = self._build_chunk_hierarchy(chunks)

        for chunk in chunks:
            # Get context from parent chunks
            parent_context = self._get_parent_context(chunk, chunk_hierarchy)
            concept_context = document.get_content()[:500]  # First 500 chars as concept context

            # Resolve coreferences
            resolution = resolver.resolve_chunk_with_context(
                chunk.content,
                parent_context,
                concept_context
            )

            # Update chunk with resolved content
            from .chunk_models import Chunk
            resolved_chunk = Chunk(
                id=chunk.id,
                content=resolution.resolved_text,
                level=chunk.level,
                parent_id=chunk.parent_id,
                child_ids=chunk.child_ids,
                metadata={
                    **chunk.metadata,
                    'coreference_applied': True,
                    'coreference_confidence': resolution.confidence_score,
                    'coreference_replacements': len(resolution.replacements)
                },
                confidence_score=min(chunk.confidence_score, resolution.confidence_score)
            )

            resolved_chunks.append(resolved_chunk)

        return resolved_chunks

    def _build_chunk_hierarchy(self, chunks: List) -> Dict[str, Any]:
        """Build hierarchy map for context retrieval"""
        hierarchy = {}
        for chunk in chunks:
            hierarchy[chunk.id] = {
                'chunk': chunk,
                'parent_id': chunk.parent_id,
                'child_ids': chunk.child_ids
            }
        return hierarchy

    def _get_parent_context(self, chunk, chunk_hierarchy: Dict[str, Any], max_length: int = 500) -> str:
        """Get parent chunk content for context"""
        if not chunk.parent_id or chunk.parent_id not in chunk_hierarchy:
            return ""

        parent_chunk = chunk_hierarchy[chunk.parent_id]['chunk']
        return parent_chunk.content[:max_length]
```

## Phase 2 Deliverables

### Week 4: NLP Integration and Multi-Format Parsing
- [ ] Coreference resolution system implementation
- [ ] PDF parser with structure detection
- [ ] HTML parser for web articles
- [ ] EPUB parser for books
- [ ] Markdown parser enhancement
- [ ] Quality assessment system

### Week 5: Integration and Quality Control
- [ ] Multi-format chunking system
- [ ] Manual review queue implementation
- [ ] Cross-document reference resolution
- [ ] Enhanced metadata schemas
- [ ] Performance optimization for multi-format processing

## Critical Decisions for Phase 2

19. **Coreference Resolution Performance**: Should coreference resolution be applied during chunking or as a post-processing step?

20. **Quality Threshold Tuning**: What confidence thresholds work best for your specific document collection?

21. **Multi-Format Priority**: Which non-org-roam formats should be prioritized for initial implementation?

## Future Capabilities (Post Phase 2)

### Advanced NLP Features
- **Named Entity Recognition**: Extract and link entities across documents
- **Relation Extraction**: Identify relationships between concepts automatically
- **Sentiment Analysis**: Classify content by sentiment for filtering
- **Topic Modeling**: Automatic topic discovery and tagging

### Enhanced Quality Control
- **Active Learning**: Use user feedback to improve quality scoring
- **Anomaly Detection**: Automatically identify unusual documents needing review
- **Quality Metrics Dashboard**: Visual interface for monitoring processing quality

### Cross-Document Intelligence
- **Duplicate Detection**: Identify similar content across documents
- **Citation Network Analysis**: Build citation graphs for academic papers
- **Content Summarization**: Generate automatic summaries for long documents
- **Cross-Reference Validation**: Verify reference accuracy and completeness

### Performance Optimization
- **Streaming Processing**: Handle large documents without loading entirely into memory
- **Distributed Processing**: Scale across multiple GPUs/machines
- **Intelligent Caching**: Cache intermediate processing results
- **Incremental Updates**: Only reprocess changed sections of documents

## Related Documentation

For conceptual background on the techniques used in this implementation:

- **Coreference Resolution**: [coreference-resolution-nlp.md](coreference-resolution-nlp.md)
- **Multi-Format Processing**: [multi-format-document-processing.md](multi-format-document-processing.md)
- **NLP Tools**: [nlp-pipeline-tools-overview.md](nlp-pipeline-tools-overview.md)
- **Chunking Strategies**: [hierarchical-document-chunking-strategies.md](hierarchical-document-chunking-strategies.md)
- **Quality Control**: See confidence scoring sections in chunking strategies

For project overview and previous phases:

- **Project Specification**: [semantic-document-processing-project-specification.md](semantic-document-processing-project-specification.md)
- **Phase 1 Implementation**: [semantic-document-processing-phase1-implementation.md](semantic-document-processing-phase1-implementation.md)

This Phase 2 implementation significantly enhances the system's capabilities while maintaining the solid foundation built in Phase 1.
