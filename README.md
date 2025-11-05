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

### Development Tools

- [development-tool-installation-strategies.md](development-tool-installation-strategies.md) - Explains why project-specific npm/pip installation prevents version conflicts and ensures team consistency.

## OLOL & Distributed LLM Inference

### Core Distributed Inference

- [olol-distributed-inference-setup.md](olol-distributed-inference-setup.md) - Describes OLOL's architecture patterns for distributing inference across multiple Ollama instances with automatic load balancing.

- [gpu-memory-model-sizing-guide.md](gpu-memory-model-sizing-guide.md) - Provides sizing formulas and tables for matching LLM models to GPU VRAM based on quantization levels and context windows.

- [multi-gpu-ollama-setup-guide.md](multi-gpu-ollama-setup-guide.md) - Explains how to run separate Ollama instances per GPU using CUDA_VISIBLE_DEVICES to maximize hardware utilization.

- [mac-studio-unified-memory-inference.md](mac-studio-unified-memory-inference.md) - Demonstrates running multiple large models simultaneously on Apple Silicon's shared memory architecture.

### Batch Processing

- [llm-batch-processing-strategies.md](llm-batch-processing-strategies.md) - Covers Python-based approaches using ThreadPoolExecutor, asyncio, and Celery for production batch inference workloads.

- [gnu-parallel-ollama-batch-processing.md](gnu-parallel-ollama-batch-processing.md) - Shows command-line batch processing using GNU parallel with curl, including job management and monitoring techniques.

## Repository Purpose

This repository serves as a knowledge base for:
- Setting up and optimizing Claude Code workflows
- Integrating AI coding assistants with Emacs
- Building distributed inference systems for LLMs
- Batch processing large volumes of inference requests

All content has been extracted from actual conversations, deduplicated, and organized into focused single-topic files for easy reference.
