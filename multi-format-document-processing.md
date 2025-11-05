# Multi-Format Document Processing

## Overview

A robust semantic search system must handle multiple document formats, each with unique structural characteristics. This guide covers strategies for processing org-mode, Markdown, PDF, HTML, and EPUB documents while preserving their hierarchical structure.

## Format-Specific Challenges

### Org-mode
- **Strengths**: Clear hierarchy with headline levels, structured properties, explicit links
- **Challenges**: Custom TODO states, non-standard properties, nested lists
- **Structure markers**: `*`, `**`, `***` for heading levels

### Markdown
- **Strengths**: Simple, widespread, clear heading structure
- **Challenges**: Inconsistent implementations, limited metadata support, linking varies
- **Structure markers**: `#`, `##`, `###` for heading levels

### PDF
- **Strengths**: Preserves visual layout, widely used for papers/books
- **Challenges**: No semantic structure, layout-based extraction, OCR issues in scanned PDFs
- **Structure markers**: Font sizes, styles, whitespace (inferred)

### HTML
- **Strengths**: Semantic markup available, hierarchical DOM structure
- **Challenges**: Varies from semantic to presentational markup, embedded scripts/styles
- **Structure markers**: `<h1>` through `<h6>`, `<section>`, `<article>`

### EPUB
- **Strengths**: HTML-based, chapter structure, metadata in manifest
- **Challenges**: Split across multiple HTML files, navigation logic
- **Structure markers**: TOC navigation document, HTML headings

## Format-Specific Parsers

### Org-mode Parser

```python
import orgparse
import re

def parse_org_mode(file_path):
    """Extract hierarchical structure from org-mode file."""

    node = orgparse.load(file_path)

    # Extract properties
    properties = {
        'id': node.get_property('ID'),
        'title': node.get_property('TITLE', node.heading),
        'aliases': node.get_property('ROAM_ALIASES', '').split(),
        'tags': node.get_tags()
    }

    # Extract hierarchy
    chunks = []

    def process_node(n, parent_path=""):
        """Recursively process org nodes."""

        path = f"{parent_path}/{n.heading}" if parent_path else n.heading
        level_map = {1: "concept", 2: "relationship", 3: "content_block"}

        chunk = {
            'text': n.get_body(format='plain'),
            'heading': n.heading,
            'level': n.level,
            'chunk_level': level_map.get(n.level, 'paragraph'),
            'hierarchy_path': path,
            'todo_state': n.todo,
            'tags': n.get_tags()
        }

        chunks.append(chunk)

        # Process children
        for child in n.children:
            process_node(child, path)

    # Process all nodes
    for child in node.children:
        process_node(child)

    # Extract org-roam links
    links = extract_org_roam_links(node)

    # Extract bibliography references
    bib_refs = extract_bibliography_refs(node)

    return {
        'properties': properties,
        'chunks': chunks,
        'links': links,
        'bibliography': bib_refs
    }

def extract_org_roam_links(node):
    """Extract org-roam style links."""

    text = node.env.filename.read_text()

    # ID-based links
    id_pattern = r'\[\[id:([^\]]+)\]\[([^\]]*)\]\]'
    id_links = re.findall(id_pattern, text)

    # File-based links
    file_pattern = r'\[\[file:([^\]]+\.org)\]\[([^\]]*)\]\]'
    file_links = re.findall(file_pattern, text)

    return {
        'id_links': id_links,
        'file_links': file_links
    }

def extract_bibliography_refs(node):
    """Extract bibliography reference markers like <23>."""

    text = node.env.filename.read_text()
    ref_pattern = r'<(\d+)>'
    refs = re.findall(ref_pattern, text)

    return list(set(refs))
```

### Markdown Parser

```python
import re

def parse_markdown(file_path):
    """Extract hierarchical structure from Markdown."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse YAML frontmatter if present
    metadata = {}
    if content.startswith('---'):
        end = content.find('---', 3)
        if end != -1:
            import yaml
            metadata = yaml.safe_load(content[3:end])
            content = content[end+3:]

    # Extract headings and content
    chunks = []
    heading_pattern = r'^(#{1,6})\s+(.+)$'

    lines = content.split('\n')
    current_chunk = {'level': 0, 'heading': 'Document', 'text': []}
    path_stack = ['Document']

    for line in lines:
        match = re.match(heading_pattern, line)

        if match:
            # Save previous chunk
            if current_chunk['text']:
                chunks.append({
                    'heading': current_chunk['heading'],
                    'level': current_chunk['level'],
                    'chunk_level': get_chunk_level(current_chunk['level']),
                    'hierarchy_path': '/'.join(path_stack),
                    'text': '\n'.join(current_chunk['text']).strip()
                })

            # Start new chunk
            level = len(match.group(1))
            heading = match.group(2)

            # Update path stack
            path_stack = path_stack[:level] + [heading]

            current_chunk = {'level': level, 'heading': heading, 'text': []}
        else:
            current_chunk['text'].append(line)

    # Save last chunk
    if current_chunk['text']:
        chunks.append({
            'heading': current_chunk['heading'],
            'level': current_chunk['level'],
            'chunk_level': get_chunk_level(current_chunk['level']),
            'hierarchy_path': '/'.join(path_stack),
            'text': '\n'.join(current_chunk['text']).strip()
        })

    return {
        'metadata': metadata,
        'chunks': chunks
    }

def get_chunk_level(heading_level):
    """Map heading level to chunk level."""
    level_map = {1: 'chapter', 2: 'section', 3: 'subsection', 4: 'paragraph'}
    return level_map.get(heading_level, 'paragraph')
```

### PDF Parser

```python
import fitz  # PyMuPDF

def parse_pdf(file_path):
    """Extract structure from PDF using font sizes and styles."""

    doc = fitz.open(file_path)
    chunks = []
    current_chunk = {'heading': '', 'text': [], 'level': 0}

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:  # Not a text block
                continue

            for line in block["lines"]:
                line_text = ""
                font_size = 0

                for span in line["spans"]:
                    line_text += span["text"]
                    font_size = max(font_size, span["size"])

                # Detect headings by font size
                if is_heading(font_size, span.get("flags", 0)):
                    # Save previous chunk
                    if current_chunk['text']:
                        chunks.append({
                            'heading': current_chunk['heading'],
                            'level': current_chunk['level'],
                            'chunk_level': get_chunk_level_from_size(font_size),
                            'text': '\n'.join(current_chunk['text']).strip(),
                            'page': page_num
                        })

                    # Start new chunk
                    current_chunk = {
                        'heading': line_text.strip(),
                        'text': [],
                        'level': get_level_from_size(font_size)
                    }
                else:
                    current_chunk['text'].append(line_text)

    # Save last chunk
    if current_chunk['text']:
        chunks.append({
            'heading': current_chunk['heading'],
            'level': current_chunk['level'],
            'chunk_level': get_chunk_level_from_size(12),
            'text': '\n'.join(current_chunk['text']).strip()
        })

    return {'chunks': chunks}

def is_heading(font_size, flags):
    """Determine if text is a heading based on font size and style."""
    # Bold flag is 16
    is_bold = flags & 16
    return font_size > 13 or (font_size > 11 and is_bold)

def get_level_from_size(font_size):
    """Map font size to heading level."""
    if font_size > 18:
        return 1
    elif font_size > 15:
        return 2
    elif font_size > 13:
        return 3
    else:
        return 4
```

### HTML Parser

```python
from bs4 import BeautifulSoup

def parse_html(file_path):
    """Extract hierarchical structure from HTML."""

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    chunks = []
    path_stack = []

    def process_element(element, level=1):
        """Recursively process HTML elements."""

        # Check if heading
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            heading_level = int(element.name[1])
            heading_text = element.get_text(strip=True)

            # Update path stack
            path_stack[:] = path_stack[:heading_level-1] + [heading_text]

            # Find content until next heading
            content = []
            for sibling in element.find_next_siblings():
                if sibling.name and sibling.name.startswith('h'):
                    break
                if sibling.name == 'p':
                    content.append(sibling.get_text(strip=True))

            if content or heading_text:
                chunks.append({
                    'heading': heading_text,
                    'level': heading_level,
                    'chunk_level': get_chunk_level(heading_level),
                    'hierarchy_path': '/'.join(path_stack),
                    'text': '\n'.join(content)
                })

    # Process all headings
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        process_element(heading)

    # If no headings found, treat as single document
    if not chunks:
        chunks.append({
            'heading': soup.find('title').get_text() if soup.find('title') else 'Document',
            'level': 1,
            'chunk_level': 'document',
            'hierarchy_path': 'Document',
            'text': soup.get_text(strip=True)
        })

    return {'chunks': chunks}
```

### EPUB Parser

```python
from ebooklib import epub
from bs4 import BeautifulSoup

def parse_epub(file_path):
    """Extract structure from EPUB file."""

    book = epub.read_epub(file_path)

    # Extract metadata
    metadata = {
        'title': book.get_metadata('DC', 'title'),
        'author': book.get_metadata('DC', 'creator'),
        'language': book.get_metadata('DC', 'language')
    }

    chunks = []
    chapter_num = 0

    # Process each document in the book
    for item in book.get_items():
        if item.get_type() == 9:  # EPUB_DOCUMENT
            chapter_num += 1

            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')

            # Extract chapter title
            chapter_title = soup.find('h1')
            if chapter_title:
                chapter_title = chapter_title.get_text(strip=True)
            else:
                chapter_title = f"Chapter {chapter_num}"

            # Extract paragraphs
            paragraphs = soup.find_all('p')
            chapter_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs])

            chunks.append({
                'heading': chapter_title,
                'level': 1,
                'chunk_level': 'chapter',
                'hierarchy_path': chapter_title,
                'text': chapter_text,
                'chapter_num': chapter_num
            })

            # Extract sections within chapter
            for heading in soup.find_all(['h2', 'h3']):
                level = int(heading.name[1])
                section_title = heading.get_text(strip=True)

                # Get content until next heading
                content = []
                for sibling in heading.find_next_siblings():
                    if sibling.name and sibling.name.startswith('h'):
                        break
                    if sibling.name == 'p':
                        content.append(sibling.get_text(strip=True))

                chunks.append({
                    'heading': section_title,
                    'level': level,
                    'chunk_level': 'section' if level == 2 else 'subsection',
                    'hierarchy_path': f"{chapter_title}/{section_title}",
                    'text': '\n'.join(content),
                    'chapter_num': chapter_num
                })

    return {
        'metadata': metadata,
        'chunks': chunks
    }
```

## Unified Processing Pipeline

### Format Detection

```python
import os

def detect_format(file_path):
    """Detect document format from extension."""

    ext = os.path.splitext(file_path)[1].lower()

    format_map = {
        '.org': 'org-mode',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.pdf': 'pdf',
        '.html': 'html',
        '.htm': 'html',
        '.epub': 'epub'
    }

    return format_map.get(ext, 'unknown')
```

### Unified Parser

```python
def parse_document(file_path):
    """Parse any supported document format."""

    doc_format = detect_format(file_path)

    parsers = {
        'org-mode': parse_org_mode,
        'markdown': parse_markdown,
        'pdf': parse_pdf,
        'html': parse_html,
        'epub': parse_epub
    }

    parser = parsers.get(doc_format)

    if not parser:
        raise ValueError(f"Unsupported format: {doc_format}")

    try:
        result = parser(file_path)
        result['format'] = doc_format
        result['file_path'] = file_path
        return result
    except Exception as e:
        return {
            'error': str(e),
            'format': doc_format,
            'file_path': file_path,
            'chunks': []
        }
```

### Confidence Scoring

```python
def assess_parsing_quality(parsed_doc):
    """Assess confidence in parsing results."""

    confidence = 1.0
    issues = []

    # Check for empty chunks
    empty_chunks = sum(1 for c in parsed_doc['chunks'] if not c.get('text', '').strip())
    if empty_chunks > 0:
        confidence -= 0.1
        issues.append(f"{empty_chunks} empty chunks")

    # Check for reasonable heading structure
    if len(parsed_doc['chunks']) == 1:
        confidence -= 0.2
        issues.append("No structure detected")

    # Check average chunk size
    avg_size = sum(len(c.get('text', '')) for c in parsed_doc['chunks']) / max(len(parsed_doc['chunks']), 1)
    if avg_size < 100:
        confidence -= 0.15
        issues.append("Chunks too small")
    elif avg_size > 5000:
        confidence -= 0.15
        issues.append("Chunks too large")

    # Format-specific checks
    if parsed_doc['format'] == 'pdf':
        # PDFs are harder to parse reliably
        confidence *= 0.85

    return {
        'confidence': max(confidence, 0.0),
        'issues': issues,
        'needs_review': confidence < 0.75
    }
```

## Handling Edge Cases

### OCR for Scanned PDFs

```python
try:
    import pytesseract
    from pdf2image import convert_from_path

    def parse_scanned_pdf(file_path):
        """Extract text from scanned PDF using OCR."""

        images = convert_from_path(file_path)
        text_chunks = []

        for page_num, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            text_chunks.append({
                'heading': f"Page {page_num + 1}",
                'level': 1,
                'chunk_level': 'page',
                'text': text,
                'page': page_num,
                'ocr': True
            })

        return {'chunks': text_chunks}
except ImportError:
    pass
```

### Comment Thread Processing

```python
def parse_comment_thread(data, max_depth=10):
    """Parse nested comment threads (e.g., Reddit, HN)."""

    chunks = []

    def process_comment(comment, depth=0, parent_path=""):
        """Recursively process nested comments."""

        if depth > max_depth:
            return

        path = f"{parent_path}/Comment-{comment['id']}" if parent_path else f"Thread-{comment['id']}"

        chunk = {
            'text': comment['text'],
            'heading': f"Comment by {comment['author']}",
            'level': depth,
            'chunk_level': 'comment',
            'hierarchy_path': path,
            'author': comment['author'],
            'timestamp': comment['timestamp'],
            'depth': depth,
            'parent_id': comment.get('parent_id')
        }

        chunks.append(chunk)

        # Process replies
        for reply in comment.get('replies', []):
            process_comment(reply, depth + 1, path)

    # Process top-level comments
    for comment in data.get('comments', []):
        process_comment(comment)

    return {'chunks': chunks}
```

## Quality Control

### Manual Review Queue

```python
def queue_for_review(parsed_doc, assessment):
    """Add low-confidence documents to review queue."""

    if assessment['needs_review']:
        review_queue.add({
            'file_path': parsed_doc['file_path'],
            'format': parsed_doc['format'],
            'confidence': assessment['confidence'],
            'issues': assessment['issues'],
            'chunks': parsed_doc['chunks']
        })
```

## Related Topics

- **Chunking Strategies**: See hierarchical-document-chunking-strategies.md
- **NLP Tools**: See nlp-pipeline-tools-overview.md
- **Org-roam Specifics**: See org-roam-zettelkasten-semantic-search.md
- **ChromaDB Storage**: See chromadb-hierarchical-storage.md
