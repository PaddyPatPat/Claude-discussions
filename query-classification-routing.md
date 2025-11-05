# Query Classification and Routing

## Overview

Query classification is the process of automatically determining query intent and routing searches to the optimal document chunk levels. For hierarchical document systems, different query types benefit from different chunk granularities.

## Query Types and Characteristics

### 1. Factual Queries

**Characteristics:**
- Seeks specific information or data points
- Often contains: "what is", "how many", "when did", "who"
- Expects short, precise answers

**Examples:**
- "What is the temperature threshold for polymer degradation?"
- "When was the study published?"
- "Who developed the algorithm?"

**Optimal Chunk Levels:** Sentence, Paragraph

**Rationale:** Facts are contained in specific sentences; broader context not needed.

### 2. Conceptual Queries

**Characteristics:**
- Seeks understanding of broad topics or ideas
- Often contains: "how does", "explain", "overview", "introduction to"
- Expects comprehensive explanations

**Examples:**
- "How does machine learning apply to healthcare?"
- "Explain the principles of distributed systems"
- "What is the relationship between X and Y?"

**Optimal Chunk Levels:** Section, Chapter

**Rationale:** Understanding requires broader context and multiple related points.

### 3. Procedural Queries

**Characteristics:**
- Seeks step-by-step instructions
- Often contains: "how to", "steps to", "configure", "install", "setup"
- Expects sequential information

**Examples:**
- "How do I configure the authentication system?"
- "Steps to install the software"
- "How to troubleshoot GPU issues?"

**Optimal Chunk Levels:** Section, Paragraph (with sequential context)

**Rationale:** Procedures span multiple steps but need to maintain order.

### 4. Comparative Queries

**Characteristics:**
- Seeks differences or similarities between entities
- Often contains: "difference between", "compare", "versus", "vs"
- Expects side-by-side information

**Examples:**
- "What are the differences between approach A and B?"
- "Compare Flycheck and Flymake"
- "MySQL vs PostgreSQL for this use case"

**Optimal Chunk Levels:** All levels with context reconstruction

**Rationale:** Comparisons may span multiple locations; need full context.

### 5. Navigational Queries

**Characteristics:**
- Seeks specific documents or concepts
- Often contains proper nouns, specific titles
- Expects exact matches

**Examples:**
- "Machine Learning Applications concept note"
- "Find the OLOL setup guide"
- "Show me the authentication documentation"

**Optimal Chunk Levels:** Document/Concept level

**Rationale:** User knows what they want; return the whole document.

## Implementation Strategies

### Approach 1: Rule-Based Classification (Simple)

```python
import re

def classify_query_rule_based(query):
    """Classify query using keyword patterns."""

    query_lower = query.lower()

    # Factual patterns
    factual_patterns = [
        r'^what is\b',
        r'^when did\b',
        r'^who\b',
        r'^how many\b',
        r'^what does\b'
    ]

    # Conceptual patterns
    conceptual_patterns = [
        r'\bhow does\b',
        r'\bexplain\b',
        r'\brelationship between\b',
        r'\boverview\b',
        r'\bprinciples\b'
    ]

    # Procedural patterns
    procedural_patterns = [
        r'\bhow to\b',
        r'\bsteps to\b',
        r'\bconfigure\b',
        r'\binstall\b',
        r'\bsetup\b',
        r'\btroubleshoot\b'
    ]

    # Comparative patterns
    comparative_patterns = [
        r'\bdifference between\b',
        r'\bcompare\b',
        r'\bversus\b',
        r'\bvs\b',
        r'\bor\b.*\?'
    ]

    # Check patterns
    if any(re.search(p, query_lower) for p in factual_patterns):
        return "factual"
    elif any(re.search(p, query_lower) for p in comparative_patterns):
        return "comparative"
    elif any(re.search(p, query_lower) for p in procedural_patterns):
        return "procedural"
    elif any(re.search(p, query_lower) for p in conceptual_patterns):
        return "conceptual"
    else:
        return "general"  # Default
```

**Advantages:**
- Simple to implement and debug
- Fast execution
- Transparent logic

**Disadvantages:**
- Limited to predefined patterns
- Misses nuanced queries
- Requires manual pattern maintenance

### Approach 2: ML-Based Classification (Advanced)

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify_query_ml(query):
    """Classify query using transformer model."""

    candidate_labels = [
        "factual question",
        "conceptual explanation",
        "procedural instructions",
        "comparative analysis",
        "navigational search"
    ]

    result = classifier(query, candidate_labels)

    # Get top classification
    top_label = result['labels'][0]
    confidence = result['scores'][0]

    # Map to query type
    label_map = {
        "factual question": "factual",
        "conceptual explanation": "conceptual",
        "procedural instructions": "procedural",
        "comparative analysis": "comparative",
        "navigational search": "navigational"
    }

    return {
        "type": label_map[top_label],
        "confidence": confidence
    }
```

**Advantages:**
- Handles nuanced queries
- Adaptable without code changes
- Higher accuracy

**Disadvantages:**
- Requires GPU for reasonable speed
- More complex setup
- Less transparent

### Approach 3: Hybrid (Recommended)

```python
def classify_query_hybrid(query):
    """Use rules first, ML for uncertain cases."""

    # Try rule-based first
    rule_result = classify_query_rule_based(query)

    if rule_result != "general":
        # Clear pattern match, use it
        return {"type": rule_result, "confidence": 0.9, "method": "rules"}
    else:
        # Uncertain, use ML
        ml_result = classify_query_ml(query)

        if ml_result["confidence"] > 0.7:
            return {"type": ml_result["type"], "confidence": ml_result["confidence"], "method": "ml"}
        else:
            # Low confidence, default to hybrid approach
            return {"type": "hybrid", "confidence": 0.5, "method": "fallback"}
```

## Query Routing Strategies

### Basic Routing

```python
def route_query(query, classification):
    """Route query to appropriate chunk levels."""

    routing_map = {
        "factual": ["sentence", "paragraph"],
        "conceptual": ["section", "chapter"],
        "procedural": ["section", "paragraph"],
        "comparative": ["all"],  # Search all levels
        "navigational": ["document"],
        "hybrid": ["paragraph", "section"]  # Middle ground
    }

    chunk_levels = routing_map.get(classification["type"], ["paragraph"])

    return {
        "chunk_levels": chunk_levels,
        "n_results": get_result_count(classification["type"]),
        "rerank": should_rerank(classification["type"])
    }
```

### Advanced Routing with Confidence

```python
def route_query_with_confidence(query, classification):
    """Adjust routing based on confidence."""

    base_routing = route_query(query, classification)

    if classification["confidence"] < 0.7:
        # Low confidence, search more broadly
        base_routing["chunk_levels"] = ["sentence", "paragraph", "section"]
        base_routing["n_results"] = base_routing["n_results"] * 2

    return base_routing
```

## Search Execution Strategies

### Strategy 1: Level-Specific Search

```python
def search_by_levels(query_embedding, chunk_levels, n_results=10):
    """Search only specified chunk levels."""

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"chunk_level": {"$in": chunk_levels}}
    )

    return results
```

### Strategy 2: Multi-Level with Ranking

```python
def search_multi_level_ranked(query_embedding, primary_level, secondary_level, n_results=10):
    """Search multiple levels with preference for primary."""

    # Search primary level
    primary_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"chunk_level": primary_level}
    )

    # Search secondary level
    secondary_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"chunk_level": secondary_level}
    )

    # Combine with boosting for primary
    combined = []
    for r in primary_results["documents"][0]:
        combined.append({"text": r, "score": r["score"] * 1.2, "level": "primary"})

    for r in secondary_results["documents"][0]:
        combined.append({"text": r, "score": r["score"], "level": "secondary"})

    # Sort by adjusted score
    combined.sort(key=lambda x: x["score"], reverse=True)

    return combined[:n_results]
```

### Strategy 3: Cascade Search

```python
def search_cascade(query_embedding, chunk_levels, target_results=10):
    """Search progressively finer levels until enough results."""

    all_results = []

    for level in chunk_levels:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=target_results - len(all_results),
            where={"chunk_level": level}
        )

        all_results.extend(results["documents"][0])

        if len(all_results) >= target_results:
            break

    return all_results[:target_results]
```

## Result Reranking

### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, initial_results):
    """Rerank results using cross-encoder for higher accuracy."""

    # Prepare query-document pairs
    pairs = [(query, doc["text"]) for doc in initial_results]

    # Score with cross-encoder
    scores = reranker.predict(pairs)

    # Combine with original scores
    for doc, cross_score in zip(initial_results, scores):
        doc["rerank_score"] = (
            0.5 * doc["similarity_score"] +
            0.5 * cross_score
        )

    # Resort
    initial_results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return initial_results
```

### Context-Aware Reranking

```python
def rerank_with_context(query, results, classification):
    """Rerank based on query type and context."""

    for result in results:
        base_score = result["similarity_score"]

        # Adjust for chunk level appropriateness
        level_bonus = get_level_bonus(result["chunk_level"], classification["type"])

        # Adjust for metadata
        metadata_bonus = 0
        if result.get("confidence_score", 1.0) > 0.9:
            metadata_bonus = 0.05

        result["final_score"] = base_score + level_bonus + metadata_bonus

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results
```

## User Interface Considerations

### Manual Override Option

```python
def search_with_override(query, auto_classify=True, manual_type=None):
    """Allow users to specify query type manually."""

    if manual_type:
        classification = {"type": manual_type, "confidence": 1.0}
    elif auto_classify:
        classification = classify_query_hybrid(query)
    else:
        classification = {"type": "hybrid", "confidence": 0.5}

    routing = route_query(query, classification)
    results = execute_search(query, routing)

    return results
```

### Exposing Classification to Users

```python
def search_with_explanation(query):
    """Return results with classification explanation."""

    classification = classify_query_hybrid(query)
    routing = route_query(query, classification)
    results = execute_search(query, routing)

    return {
        "results": results,
        "explanation": {
            "query_type": classification["type"],
            "confidence": classification["confidence"],
            "searched_levels": routing["chunk_levels"],
            "rationale": get_rationale(classification["type"])
        }
    }
```

## Performance Monitoring

### Track Classification Accuracy

```python
query_log = {
    "query": user_query,
    "classification": classification,
    "user_satisfaction": None,  # To be filled by feedback
    "results_count": len(results),
    "top_result_level": results[0]["chunk_level"]
}

# Later, analyze patterns
def analyze_classification_performance(logs):
    """Analyze which classifications perform well."""

    satisfied_by_type = {}
    for log in logs:
        if log["user_satisfaction"] is not None:
            qtype = log["classification"]["type"]
            satisfied_by_type.setdefault(qtype, []).append(log["user_satisfaction"])

    for qtype, satisfactions in satisfied_by_type.items():
        avg_satisfaction = sum(satisfactions) / len(satisfactions)
        print(f"{qtype}: {avg_satisfaction:.2f}")
```

## Org-roam Specific Routing

### Zettelkasten Query Types

```python
def classify_zettelkasten_query(query):
    """Special classification for org-roam queries."""

    query_lower = query.lower()

    # Concept discovery
    if any(word in query_lower for word in ["about", "concept", "know about"]):
        return {"type": "concept_discovery", "level": "concept"}

    # Relationship queries
    if any(word in query_lower for word in ["requires", "leads to", "outcome", "prerequisite"]):
        return {"type": "relationship", "level": "relationship"}

    # Source-grounded
    if "<" in query and ">" in query:  # Bibliography ref like <23>
        return {"type": "source_grounded", "level": "content_block"}

    # Default to standard classification
    return classify_query_hybrid(query)
```

## Related Topics

- **Chunking Strategies**: See hierarchical-document-chunking-strategies.md
- **Embedding Selection**: See semantic-search-embedding-strategies.md
- **ChromaDB Implementation**: See chromadb-hierarchical-storage.md
- **Org-roam Specifics**: See org-roam-zettelkasten-semantic-search.md
