# NLP Pipeline Tools Overview

## Overview

Building a semantic document processing system requires multiple NLP tools working together. This guide covers the essential tools for text preprocessing, coreference resolution, embeddings, and document parsing.

## Core NLP Libraries

### spaCy

**Purpose**: Industrial-strength NLP for production use

**Key Features:**
- Fast and efficient (Cython-based)
- Pre-trained models for multiple languages
- Tokenization, POS tagging, named entity recognition
- Dependency parsing
- Extensible pipeline architecture

**Installation:**
```bash
pip install spacy
python -m spacy download en_core_web_lg  # Large English model
```

**Basic Usage:**
```python
import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Machine learning is transforming healthcare.")

# Tokenization
for token in doc:
    print(token.text, token.pos_, token.dep_)

# Named entities
for ent in doc.ents:
    print(ent.text, ent.label_)

# Sentence segmentation
for sent in doc.sents:
    print(sent.text)
```

**When to Use:**
- Production document processing
- Need for speed and efficiency
- Require multiple NLP tasks in one pipeline
- Building custom pipelines with extensions

**Limitations:**
- Less accurate than transformer models for some tasks
- English-centric (though multi-language support exists)

### Coreferee (spaCy Extension)

**Purpose**: Coreference resolution for spaCy

**Installation:**
```bash
pip install coreferee
python -m coreferee install en
```

**Usage:**
```python
import spacy
import coreferee

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")

doc = nlp("Dave likes to walk. He walks in the park.")

# Access coreference chains
for chain in doc._.coref_chains:
    print(f"Chain: {chain}")
    for mention in chain:
        print(f"  - {mention.text}")

# Resolve text
resolved = doc._.coref_chains.resolve(doc)
print(resolved)
```

**When to Use:**
- Fast coreference resolution needed
- Integration with existing spaCy pipelines
- Good enough accuracy for general text

**Limitations:**
- Less accurate than transformer-based models
- May struggle with complex literary text

### AllenNLP

**Purpose**: Research-grade NLP with state-of-the-art models

**Key Features:**
- Transformer-based models
- High-accuracy coreference resolution
- Pre-built predictors for common tasks
- PyTorch-based

**Installation:**
```bash
pip install allennlp allennlp-models
```

**Coreference Resolution:**
```python
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
)

text = "Dave likes to walk. He walks in the park."
result = predictor.predict(document=text)

# Extract coreference clusters
clusters = result["clusters"]
print(f"Coreference clusters: {clusters}")

# Resolved text
def resolve_coreferences(text, clusters, tokens):
    """Replace pronouns with their referents."""
    # Implementation of resolution logic
    resolved = text  # Simplified
    return resolved
```

**When to Use:**
- Need highest accuracy for coreference
- Research or high-value documents
- GPU available for processing

**Limitations:**
- Slower than spaCy
- Requires more computational resources
- More complex setup

## Embedding and Semantic Search Libraries

### Sentence Transformers

**Purpose**: Generate embeddings for semantic search

**Installation:**
```bash
pip install sentence-transformers
```

**Basic Usage:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

# Single sentence
embedding = model.encode("This is a test sentence")

# Batch encoding
sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
embeddings = model.encode(sentences, batch_size=32)

# Similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embedding1, embedding2)
```

**When to Use:**
- Need to generate embeddings for search
- Building semantic search systems
- Comparing text similarity

**Available Models:**
- `all-MiniLM-L6-v2`: Fast, 384 dimensions
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `multi-qa-mpnet-base-dot-v1`: Optimized for Q&A

### Hugging Face Transformers

**Purpose**: Access to thousands of pre-trained models

**Installation:**
```bash
pip install transformers torch
```

**Usage:**
```python
from transformers import pipeline

# Zero-shot classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a query about machine learning",
    candidate_labels=["factual", "conceptual", "procedural"]
)

# Question answering
qa = pipeline("question-answering")
result = qa(
    question="What is machine learning?",
    context="Machine learning is a subset of AI that enables systems to learn from data."
)
```

**When to Use:**
- Need specialized models (classification, QA, etc.)
- Experimenting with latest research models
- Fine-tuning on custom data

## Document Parsing Libraries

### orgparse (Org-mode)

**Purpose**: Parse org-mode files

**Installation:**
```bash
pip install orgparse
```

**Usage:**
```python
import orgparse

node = orgparse.load('zettelkasten/concept.org')

# Access properties
print(node.get_property('ID'))
print(node.get_property('ROAM_ALIASES'))

# Iterate through headings
for child in node.children:
    print(f"Heading: {child.heading}")
    print(f"Level: {child.level}")
    print(f"TODO state: {child.todo}")
    print(f"Body: {child.get_body()}")

# Extract all headings of a specific name
requirements = [h for h in node[1:] if h.heading == "Requirements"]
```

**When to Use:**
- Processing org-roam notes
- Extracting structured information from org files
- Building org-mode-based knowledge systems

### PyMuPDF (fitz)

**Purpose**: PDF parsing and structure extraction

**Installation:**
```bash
pip install PyMuPDF
```

**Usage:**
```python
import fitz  # PyMuPDF

doc = fitz.open("document.pdf")

for page_num, page in enumerate(doc):
    # Extract text
    text = page.get_text()

    # Extract with layout information
    blocks = page.get_text("dict")["blocks"]

    # Extract by structure
    for block in blocks:
        if block["type"] == 0:  # Text block
            print(f"Block: {block['bbox']}")  # Bounding box
            for line in block["lines"]:
                for span in line["spans"]:
                    print(f"Text: {span['text']}")
                    print(f"Font: {span['font']}, Size: {span['size']}")

    # Extract images
    images = page.get_images()
```

**When to Use:**
- Extracting text from PDFs
- Need layout/structure information
- Processing academic papers or technical documents

**Alternatives:**
- **pdfplumber**: Better for tables and structured data
- **PyPDF2**: Simpler but less powerful

### python-docx (Word Documents)

**Installation:**
```bash
pip install python-docx
```

**Usage:**
```python
from docx import Document

doc = Document('document.docx')

# Extract paragraphs
for para in doc.paragraphs:
    print(para.text)
    print(para.style.name)  # Heading 1, Normal, etc.

# Extract tables
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            print(cell.text)
```

### BeautifulSoup (HTML)

**Installation:**
```bash
pip install beautifulsoup4 lxml
```

**Usage:**
```python
from bs4 import BeautifulSoup

with open('document.html', 'r') as f:
    soup = BeautifulSoup(f, 'lxml')

# Extract by tags
headings = soup.find_all(['h1', 'h2', 'h3'])
paragraphs = soup.find_all('p')

# Extract with hierarchy
for section in soup.find_all('section'):
    heading = section.find('h2')
    content = section.find_all('p')
    print(f"Section: {heading.text if heading else 'Unknown'}")
    for p in content:
        print(f"  {p.text}")

# Clean text
text = soup.get_text(strip=True, separator=' ')
```

### ebooklib (EPUB)

**Installation:**
```bash
pip install ebooklib
```

**Usage:**
```python
from ebooklib import epub
from bs4 import BeautifulSoup

book = epub.read_epub('book.epub')

for item in book.get_items():
    if item.get_type() == 9:  # EPUB document type
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        text = soup.get_text()
        print(text)
```

### Pandoc (Universal Converter)

**Installation:**
```bash
# System-level installation required
sudo apt install pandoc  # Linux
brew install pandoc      # Mac

# Python wrapper
pip install pypandoc
```

**Usage:**
```python
import pypandoc

# Convert markdown to org-mode
output = pypandoc.convert_file('document.md', 'org')

# Convert with options
output = pypandoc.convert_file(
    'document.docx',
    'markdown',
    extra_args=['--extract-media=./media']
)
```

**When to Use:**
- Need to convert between formats
- Want to preserve structure during conversion
- Processing multiple document types

## Vector Database

### ChromaDB

**Purpose**: Store and search embeddings

**Installation:**
```bash
pip install chromadb
```

**Usage:**
```python
import chromadb

client = chromadb.Client()

# Create collection
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=["Document 1 text", "Document 2 text"],
    metadatas=[{"type": "paragraph"}, {"type": "section"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["machine learning"],
    n_results=10,
    where={"type": "paragraph"}
)
```

**When to Use:**
- Need vector search capabilities
- Building semantic search systems
- Want simple, embedded database

**Alternatives:**
- **Pinecone**: Cloud-based, scalable
- **Weaviate**: More features, GraphQL API
- **Milvus**: High performance, distributed

## Putting It Together

### Complete Pipeline Example

```python
import spacy
import coreferee
from sentence_transformers import SentenceTransformer
import chromadb
import orgparse

# Initialize tools
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("org_roam_notes")

def process_org_roam_file(file_path):
    """Complete processing pipeline."""

    # 1. Parse org file
    node = orgparse.load(file_path)
    concept_id = node.get_property('ID')
    concept_title = node.get_property('TITLE', '')

    # 2. Extract sections
    for heading in node.children:
        section_text = heading.get_body()

        # 3. Apply coreference resolution
        doc = nlp(section_text)
        resolved_text = resolve_text(doc)

        # 4. Create chunks
        chunks = create_chunks(resolved_text, heading.heading)

        # 5. Generate embeddings
        embeddings = embedding_model.encode([c['text'] for c in chunks])

        # 6. Store in ChromaDB
        collection.add(
            documents=[c['text'] for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{
                'concept_id': concept_id,
                'concept_title': concept_title,
                'relationship_type': heading.heading,
                'chunk_level': c['level']
            } for c in chunks],
            ids=[c['id'] for c in chunks]
        )

# Process all org-roam files
for file_path in find_org_roam_files():
    process_org_roam_file(file_path)
```

## Performance Considerations

### Processing Speed Comparison

**Text Processing (1000 documents, avg 500 words):**
- spaCy (CPU): ~2-5 minutes
- AllenNLP (CPU): ~30-60 minutes
- AllenNLP (GPU): ~5-15 minutes

**Embedding Generation (1000 documents, avg 500 words):**
- sentence-transformers (CPU): ~10-30 minutes
- sentence-transformers (GPU): ~1-3 minutes

### Resource Requirements

**Memory:**
- spaCy: 500MB - 1GB
- AllenNLP models: 2-4GB
- Embedding models: 500MB - 2GB
- ChromaDB: Minimal (+ storage for embeddings)

**GPU:**
- Recommended for: AllenNLP, sentence-transformers
- Speedup: 10-50× depending on task

## Related Topics

- **Coreference Resolution**: See coreference-resolution-nlp.md
- **Embedding Strategies**: See semantic-search-embedding-strategies.md
- **Chunking Implementation**: See hierarchical-document-chunking-strategies.md
- **Multi-Format Processing**: See multi-format-document-processing.md
