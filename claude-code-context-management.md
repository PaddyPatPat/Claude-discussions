# Claude Code Context Window Management

## Understanding Context Usage

If you're at 12% left until auto-compaction, you've used approximately 88% of your context window in Claude Code.

## Effective Strategies to Manage Context Window

### Immediate Actions

- **Clean up unnecessary files** - Remove any test files, logs, or temporary files you don't need Claude to track
- **Archive completed work** - Move finished features or resolved issues to separate directories that Claude doesn't need to monitor
- **Use .gitignore or .claudeignore** - Exclude large files, dependencies, build outputs, and other files Claude doesn't need to see

### Ongoing Practices

- **Break down large tasks** - Instead of one massive refactor, split it into smaller, focused sessions
- **Be selective about file inclusion** - Only include files that are directly relevant to your current task
- **Use concise commit messages and comments** - This helps Claude understand context without verbose explanations
- **Summarize previous work** - When starting a new session, give Claude a brief summary of what's been accomplished rather than relying entirely on conversation history

### Advanced Techniques

- **Create documentation snapshots** - Write brief summaries of major decisions or architectural choices in a README, then start fresh sessions referencing these docs
- **Use Claude Code's reset functionality** - Start a new session when switching to unrelated tasks
- **Modularize your codebase** - Well-organized, modular code requires less context for Claude to understand individual components

## Auto-Compaction

The auto-compaction feature will help manage this automatically, but the strategies above will help you work more efficiently within the context limits.
