# Claude Discussions Repository

A collection of educational content extracted from conversations with Claude, organized into focused topic files covering Claude Code, Emacs integration, and distributed LLM inference.

## Claude Code & Emacs Integration

### Core Claude Code Topics

- [claude-code-session-timing.md](claude-code-session-timing.md) - Explains why the 5-hour limit warning may appear earlier than expected and how to troubleshoot timing discrepancies.

- [claude-code-context-management.md](claude-code-context-management.md) - Strategies for managing the context window efficiently, including file cleanup, task breakdown, and auto-compaction features.

- [claude-code-conversation-transfer.md](claude-code-conversation-transfer.md) - Details the limitations of transferring conversations between Claude web app and Claude Code, plus workaround options.

- [claude-code-installation-methods.md](claude-code-installation-methods.md) - Compares npm versus native binary installation, recommending native for better performance and automatic updates.

- [claude-code-privacy-context-control.md](claude-code-privacy-context-control.md) - Details what information gets sent to Claude automatically versus manually, with strategies for different privacy levels.

### Emacs Integration

- [claude-code-emacs-configuration.md](claude-code-emacs-configuration.md) - Complete production-ready setup using straight.el with Flycheck, vterm, Monet, and comprehensive linting configuration.

- [emacs-claude-code-package-comparison.md](emacs-claude-code-package-comparison.md) - Compares manzaltu/claude-code-ide.el versus stevemolitor/claude-code.el, recommending the latter for maturity and features.

- [monet-emacs-ide-integration.md](monet-emacs-ide-integration.md) - Explains the WebSocket-based IDE protocol that enables automatic context sharing, live diagnostics, and interactive diff views.

- [claude-code-error-fixing-workflow.md](claude-code-error-fixing-workflow.md) - Demonstrates how the `C-c c e` command detects errors from Flycheck/Flymake and sends them with context to Claude.

- [claude-code-project-activation.md](claude-code-project-activation.md) - Shows how to enable Claude Code only for approved projects using project.el or directory-based detection patterns.

- [flycheck-vs-flymake-comparison.md](flycheck-vs-flymake-comparison.md) - Recommends Flycheck for learners due to superior error messages, multi-checker support, and educational feedback quality.

### Emacs Org-Roam & Git Integration

- [emacs-orgmode-git-content-tracking.org](emacs-orgmode-git-content-tracking.org) - Tracks content movements between org-roam files in Zettelkasten workflows using magit, addressing the challenge of committing related file changes.

- [git-move-detection-analysis.org](git-move-detection-analysis.org) - Analyzes Git's built-in move detection (`-M`, `-C` flags) and explains why file-level similarity metrics are unsuitable for detecting small content block movements.

- [emacs-text-comparison-functions.org](emacs-text-comparison-functions.org) - Catalogs Emacs text similarity functions (`compare-windows`, `ediff`, `diff-buffers`) and algorithms (LCS, Levenshtein) for hunk-level comparison.

- [magit-custom-sections-integration.org](magit-custom-sections-integration.org) - Explains magit's section-based architecture, custom section integration via hooks, and direct buffer parsing strategies for accessing diff content.

- [orgmode-content-similarity-detection.org](orgmode-content-similarity-detection.org) - Designs similarity detection algorithms with org-mode normalization (heading levels, whitespace), token-based comparison, and 85% threshold rationale.

- [magit-buffer-parsing-debugging.org](magit-buffer-parsing-debugging.org) - Chronicles iterative prototype development, debugging techniques, common pitfalls, and lessons learned from parsing magit-status buffers.

#### Magit Move Detection Project

- [magit-orgmode-move-detection-project.org](magit-orgmode-move-detection-project.org) - Complete project overview for detecting content moves between org-roam files, with goals, scope, design decisions, and success criteria.

- [magit-move-detection-requirements.org](magit-move-detection-requirements.org) - Detailed functional requirements including content move definitions, detection timing, hunk grouping rules, and user interface specifications.

- [magit-move-detection-technical-architecture.org](magit-move-detection-technical-architecture.org) - Technical implementation details covering buffer parsing, hunk grouping, Levenshtein distance algorithm, and performance considerations.

- [magit-move-detection-similarity-algorithms.org](magit-move-detection-similarity-algorithms.org) - Comprehensive comparison of text similarity algorithms (prefix matching, Levenshtein, token-based, hybrid) with examples and performance analysis.

- [magit-move-detection-prototype-evolution.org](magit-move-detection-prototype-evolution.org) - Development log tracking versions 2.3 through 2.5, documenting bugs found, improvements made, and lessons learned through user testing.

- [magit-move-detection.el](magit-move-detection.el) - Working Elisp prototype (v2.5) implementing multi-line hunk grouping, Levenshtein similarity matching, and org-mode normalization.

### Development Tools

- [development-tool-installation-strategies.md](development-tool-installation-strategies.md) - Explains why project-specific npm/pip installation prevents version conflicts and ensures team consistency.

- [npm-nodejs-maintenance-guide.md](npm-nodejs-maintenance-guide.md) - Covers updating npm/Node.js, using nvm for version management, understanding version isolation, and troubleshooting with npm doctor.

## Network Storage & Backup

### TrueNAS Integration

- [truenas-mac-rsync-ssh-setup.md](truenas-mac-rsync-ssh-setup.md) - Complete guide for setting up SSH key-based authentication between macOS and TrueNAS for secure, password-less Rsync operations, including SSH config simplification, automated backup scripts, and security best practices.

## OLOL & Distributed LLM Inference

### Core Distributed Inference

- [olol-distributed-inference-setup.md](olol-distributed-inference-setup.md) - Describes OLOL's architecture patterns for distributing inference across multiple Ollama instances with automatic load balancing.

- [gpu-memory-model-sizing-guide.md](gpu-memory-model-sizing-guide.md) - Provides sizing formulas and tables for matching LLM models to GPU VRAM based on quantization levels and context windows.

- [multi-gpu-ollama-setup-guide.md](multi-gpu-ollama-setup-guide.md) - Explains how to run separate Ollama instances per GPU using CUDA_VISIBLE_DEVICES to maximize hardware utilization.

- [mac-studio-unified-memory-inference.md](mac-studio-unified-memory-inference.md) - Demonstrates running multiple large models simultaneously on Apple Silicon's shared memory architecture.

- [ollama-gpu-troubleshooting-guide.md](ollama-gpu-troubleshooting-guide.md) - Diagnoses why Ollama uses CPU instead of GPU, covers CUDA toolkit installation, port conflicts, and environment configuration.

### Batch Processing

- [llm-batch-processing-strategies.md](llm-batch-processing-strategies.md) - Covers Python-based approaches using ThreadPoolExecutor, asyncio, and Celery for production batch inference workloads.

- [gnu-parallel-ollama-batch-processing.md](gnu-parallel-ollama-batch-processing.md) - Shows command-line batch processing using GNU parallel with curl, including job management and monitoring techniques.

## Knowledge Management & Semantic Search

### Project Specification

- [semantic-document-processing-project-specification.md](semantic-document-processing-project-specification.md) - Complete project specification for building a semantic document processing system with hierarchical chunking, org-roam integration, and multi-level embeddings.

### Implementation Guides

- [semantic-document-processing-phase1-implementation.md](semantic-document-processing-phase1-implementation.md) - Phase 1 foundation implementation with complete Python code for org-roam parser, zettelkasten chunker, reference extractor, and ChromaDB client.

- [semantic-document-processing-phase2-implementation.md](semantic-document-processing-phase2-implementation.md) - Phase 2 NLP enhancement with coreference resolution, multi-format parsers (PDF, HTML, EPUB), quality control system, and confidence scoring.

- [semantic-document-processing-phase3-implementation.md](semantic-document-processing-phase3-implementation.md) - Phase 3 advanced features with multi-model embeddings, intelligent query classification/routing, advanced search engine, and context reconstruction.

### Semantic Document Processing

- [hierarchical-document-chunking-strategies.md](hierarchical-document-chunking-strategies.md) - Recursive document splitting at multiple structural levels (chapter→section→paragraph→sentence) optimized for different query types.

- [coreference-resolution-nlp.md](coreference-resolution-nlp.md) - Resolving pronouns and references before chunking to preserve context when text is split into embeddings.

- [semantic-search-embedding-strategies.md](semantic-search-embedding-strategies.md) - Multi-level embedding generation with model selection, chunk size optimization, and GPU acceleration for semantic search.

- [query-classification-routing.md](query-classification-routing.md) - Automatic detection of query intent (factual, conceptual, procedural) to route searches to optimal chunk sizes.

### Knowledge Graph Integration

- [org-roam-zettelkasten-semantic-search.md](org-roam-zettelkasten-semantic-search.md) - Integrating org-roam concept notes with semantic search, preserving bidirectional links and relationship types.

- [chromadb-hierarchical-storage.md](chromadb-hierarchical-storage.md) - Vector database storage strategies for multi-level document chunks with parent-child relationship tracking.

### Document Processing Tools

- [nlp-pipeline-tools-overview.md](nlp-pipeline-tools-overview.md) - Comprehensive guide to spaCy, AllenNLP, sentence-transformers, and document parsers for building NLP pipelines.

- [multi-format-document-processing.md](multi-format-document-processing.md) - Format-specific parsers and unified processing for org-mode, Markdown, PDF, HTML, and EPUB documents.

## NLP & Student Argument Analysis

### NLP Tools & Libraries

- [nlp-tools-comparison.org](nlp-tools-comparison.org) - Comprehensive comparison of NLP libraries (spaCy, NLTK, Transformers, Stanza, Gensim) with use cases, code examples, and recommendations by experience level.

### Argument Mining Workflow

- [nlp-student-argument-analysis.org](nlp-student-argument-analysis.org) - Complete methodology for analyzing student argumentative essays, including iterative NLP workflow, agreement/disagreement detection, and comparative discourse analysis techniques.

### Infrastructure & Integration

- [nlp-mcp-server-integration.org](nlp-mcp-server-integration.org) - MCP (Model Context Protocol) server architecture for exposing NLP tools, with implementation examples for document analysis, argument extraction, clustering, and disagreement detection.

- [nlp-document-provenance-tracking.md](nlp-document-provenance-tracking.md) - Maintaining connections between source documents and NLP analysis results across multiple iterations using content hashing, SQLite/ChromaDB storage, and version tracking strategies.

- [nlp-vector-store-integration.md](nlp-vector-store-integration.md) - Integrating ChromaDB and FAISS for storing and querying document embeddings, with performance comparisons, hybrid approaches, and hierarchical storage patterns.

- [nlp-hierarchical-embeddings.md](nlp-hierarchical-embeddings.md) - Generating multi-level embeddings (document/section/paragraph/sentence), query routing strategies, cross-level analysis, and use cases for finding similar arguments and evidence extraction.

## Repository Purpose

This repository serves as a knowledge base for:
- Setting up and optimizing Claude Code workflows
- Integrating AI coding assistants with Emacs
- Building org-roam and magit extensions for Zettelkasten workflows
- Developing Elisp tools for git content tracking and similarity detection
- Configuring secure SSH/Rsync connections to TrueNAS for automated backups
- Building distributed inference systems for LLMs
- Batch processing large volumes of inference requests
- Implementing semantic search and knowledge management systems
- Processing documents for vector embeddings and retrieval
- Analyzing student argumentative essays with NLP techniques
- Building MCP servers for NLP pipelines and document analysis

All content has been extracted from actual conversations, deduplicated, and organized into focused single-topic files for easy reference.
