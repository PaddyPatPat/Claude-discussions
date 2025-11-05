# Coreference Resolution for NLP

## What is Coreference Resolution?

Coreference resolution is the task of identifying when different expressions in text refer to the same entity. It's a critical preprocessing step for document chunking because it maintains semantic meaning when text is split into smaller pieces.

## The Problem

### Example 1: Pronoun Resolution
**Before resolution:**
```
Dave likes to walk. He walks in the park.
```
**Issue**: If split into separate chunks, "He walks in the park" loses its subject reference.

**After resolution:**
```
Dave likes to walk. Dave walks in the park.
```
**Benefit**: Each sentence maintains complete meaning independently.

### Example 2: Complex Anaphora
**Before resolution:**
```
The neural network achieved 95% accuracy. This result exceeded expectations.
The model was trained for 3 days.
```

**After resolution:**
```
The neural network achieved 95% accuracy. The neural network's 95% accuracy result
exceeded expectations. The neural network was trained for 3 days.
```

## Related NLP Concepts

### Anaphora Resolution
A subset of coreference resolution focusing specifically on backward references (anaphora):
- Pronouns: he, she, it, they
- Demonstratives: this, that, these, those
- Definite articles: the network, the system

### Entity Linking
Connecting mentions to specific entities across a document or knowledge base.

## Why This Matters for Semantic Search

### Context Preservation
When documents are chunked for embeddings:
- Small chunks (sentences) lose critical context without coreference resolution
- Search results may be ambiguous or meaningless
- Embedding quality degrades without explicit entity references

### Example Impact on Retrieval
**Query**: "What does Dave do in the park?"

**Without coreference resolution:**
- Chunk 1: "Dave likes to walk."
- Chunk 2: "He walks in the park." ❌ No match (doesn't mention "Dave")

**With coreference resolution:**
- Chunk 1: "Dave likes to walk."
- Chunk 2: "Dave walks in the park." ✅ Matches query

## Implementation Tools

### spaCy with Coreference Extensions
```python
import spacy
import coreferee

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")

doc = nlp("Dave likes to walk. He walks in the park.")
# Access coreference chains
for chain in doc._.coref_chains:
    print(chain)
```

**Advantages:**
- Fast and efficient
- Integrates with spaCy's pipeline
- Good for production use

**Limitations:**
- May miss complex coreferences
- Works best with well-formed text

### AllenNLP Coreference Resolution
```python
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
)

result = predictor.predict(
    document="Dave likes to walk. He walks in the park."
)
```

**Advantages:**
- State-of-the-art accuracy
- Handles complex linguistic phenomena
- Research-grade models

**Limitations:**
- Slower than spaCy
- Higher computational requirements

### Hugging Face Transformers
```python
from transformers import pipeline

coref = pipeline("coreference-resolution")
result = coref("Dave likes to walk. He walks in the park.")
```

**Advantages:**
- Access to latest transformer models
- Continually updated with new research
- Good multilingual support

**Limitations:**
- Requires GPU for reasonable performance
- Model size can be large

## Processing Strategy

### When to Apply Coreference Resolution

**Before Chunking (Recommended)**
```
Document → Coreference Resolution → Hierarchical Chunking → Embeddings
```
**Advantages:**
- Chunks are self-contained
- Better embedding quality
- Simpler retrieval logic

**After Chunking (Not Recommended)**
- Coreferences may span chunk boundaries
- Difficult to resolve without full context
- May require chunk merging and re-splitting

### Scope of Resolution

**Document-Level Resolution**
- Resolve all coreferences within entire document
- Best for narrative text, articles, papers
- May be computationally expensive for very long documents

**Section-Level Resolution**
- Resolve within major sections independently
- Faster processing for large documents
- Risk: May miss cross-section coreferences

**Paragraph-Level Resolution**
- Minimal context window
- Fastest processing
- Risk: Misses many coreferences

**Recommended**: Use document-level for documents under 10,000 tokens; section-level for larger documents.

## Confidence Scoring

Most coreference resolution systems provide confidence scores for their predictions:

```python
{
    "mention": "He",
    "referent": "Dave",
    "confidence": 0.87
}
```

### Using Confidence Scores
- **High confidence (>0.85)**: Apply resolution automatically
- **Medium confidence (0.65-0.85)**: Flag for manual review or apply with metadata annotation
- **Low confidence (<0.65)**: Skip resolution; preserve original text

## Special Considerations

### Technical Documents
- May have specialized referring expressions
- Acronyms and abbreviations need special handling
- Example: "The API returns JSON. This format is widely supported."

### Multi-Language Documents
- Coreference patterns differ across languages
- Pronoun systems vary (e.g., gender-neutral pronouns)
- Requires language-specific models

### Dialogue and Comments
- Speaker attribution is critical
- May need conversation structure awareness
- Example: "User A: I love this feature. User B: It's amazing!" (both "it" and "this feature" refer to same entity)

## Performance Optimization

### For Large Document Collections

1. **Batch Processing**: Process multiple documents in parallel
2. **GPU Acceleration**: Use GPU for transformer-based models
3. **Caching**: Cache resolved entities for documents that change infrequently
4. **Hybrid Approach**: Use fast spaCy for most documents; advanced models for critical documents

### Performance Benchmarks (Approximate)
- spaCy with coreferee: ~100-500 documents/hour (CPU)
- AllenNLP: ~20-100 documents/hour (GPU)
- Transformer models: ~10-50 documents/hour (GPU)

*Note: Speeds vary significantly based on document length and hardware.*

## Quality Control

### Validation Strategies
1. **Sample Testing**: Manually review resolutions on sample documents
2. **Confidence Thresholds**: Only apply high-confidence resolutions
3. **Error Detection**: Flag unusual patterns (e.g., too many replacements)
4. **A/B Testing**: Compare search quality with and without resolution

### Common Errors
- **Over-resolution**: Replacing too many pronouns, creating awkward text
- **Wrong Referent**: "It" resolved to incorrect entity
- **Lost Nuance**: "This approach" becoming "Machine learning approach" loses specificity

## Integration with Document Processing Pipeline

### Complete Workflow
```
1. Document Parsing (format-specific)
   ↓
2. Structure Detection (chapters, sections, paragraphs)
   ↓
3. Coreference Resolution ← (Critical step)
   ↓
4. Hierarchical Chunking
   ↓
5. Embedding Generation
   ↓
6. Vector Database Storage
```

## Related Topics

- **Chunking Implementation**: See hierarchical-document-chunking-strategies.md
- **NLP Tools**: See nlp-pipeline-tools-overview.md
- **Embedding Strategies**: See semantic-search-embedding-strategies.md
- **Org-roam Specifics**: See org-roam-zettelkasten-semantic-search.md
