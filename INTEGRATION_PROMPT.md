# Claude Code Integration Session Prompt

Use this prompt to start a new Claude Code session for integrating content into the Claude-discussions repository.

---

## Your Role

You are integrating conversation content into the **Claude-discussions** repository, which stores Claude AI conversations as deduplicated, focused topic files. Your task is to process the provided content and create well-organized markdown files that fit the repository's structure.

## Repository Context

**Repository**: PaddyPatPat/Claude-discussions
**Purpose**: Knowledge base of educational content from Claude conversations
**Current Topics**: Claude Code, Emacs integration, distributed LLM inference, semantic search systems

**File Structure**:
- All topic files are markdown (.md) at repository root
- README.md contains organized index with descriptions
- Files use kebab-case naming: `topic-subtopic-description.md`
- Each file focuses on ONE specific concept or implementation

## Integration Workflow

### 1. Receive Content
The user will provide content in one of these formats:
- Full conversation transcript
- Individual artifact files (specifications, implementation guides, etc.)
- Conceptual documentation
- Code examples with explanations

### 2. Analyze Content
Before creating files, identify:
- **Main topics** - What are the core concepts?
- **Natural divisions** - Should this be 1 file or multiple?
- **Content type** - Is this conceptual explanation, implementation guide, comparison, or specification?
- **Existing overlap** - Quickly grep for related keywords to avoid duplication

### 3. Create Files

**File Naming Rules**:
- Use kebab-case: `hierarchical-document-chunking-strategies.md`
- Be descriptive but concise (3-6 words typical)
- Lead with the main topic keyword for related files
- Examples:
  - `semantic-document-processing-project-specification.md`
  - `semantic-document-processing-phase1-implementation.md`
  - `claude-code-emacs-configuration.md`

**Content Organization**:
- **Conceptual files**: Explain techniques, compare approaches, provide overview
- **Implementation files**: Include complete code, step-by-step guides, working examples
- **Specification files**: Define requirements, architecture, project structure
- **Guide files**: How-to instructions, troubleshooting, configuration

**Quality Standards**:
- Minimum 200 lines for substantial topics (exceptions for narrow focused topics)
- Include code examples where relevant
- Add "Related Documentation" section linking to related files
- Use proper markdown formatting (headers, code blocks, lists)
- Preserve technical accuracy from source conversation

### 4. Update README

After creating files, update README.md:

**Section Placement**:
- Determine which major section fits (or create new section if needed)
- Create subsections as needed for logical grouping
- Maintain existing organizational patterns

**Entry Format**:
```markdown
- [filename.md](filename.md) - One sentence description that adds value beyond the filename.
```

**Description Guidelines**:
- Start with action verb or description of what the file covers
- Be specific about content (mention key technologies, approaches, or outcomes)
- Don't just restate the filename
- Examples:
  - ✅ "Phase 1 foundation implementation with complete Python code for org-roam parser, zettelkasten chunker, reference extractor, and ChromaDB client."
  - ❌ "Phase 1 implementation guide"

### 5. Git Workflow

**Branch Management**:
- Work on branch: `claude/{task-description}-{sessionID}`
- Session ID is provided in environment context
- Example: `claude/explore-repo-structure-011CUpaCH5vCA5eqnwwXEAPN`
- Create branch if it doesn't exist: `git checkout -b {branch-name}`

**Commit Strategy**:
- Commit after each logical unit (typically after creating 1-3 related files + README update)
- Use descriptive commit messages:
  - ✅ "Add Phase 1 implementation guide for semantic document processing"
  - ✅ "Add semantic search and knowledge management topic files"
  - ❌ "Update files"

**Pushing**:
- Push to remote after each commit: `git push -u origin {branch-name}`
- If push fails with network error, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s)
- Verify push success before proceeding

### 6. Deduplication Strategy

**Check for existing content**:
```bash
# Search for related keywords
grep -r "keyword" *.md --files-with-matches

# Find similar filenames
ls -1 | grep "pattern"
```

**When overlap detected**:
- If substantial overlap (>70%): Enhance existing file rather than creating new one
- If partial overlap (30-70%): Create new file with cross-references
- If minimal overlap (<30%): Create new file independently

**Cross-referencing**:
Add "Related Documentation" or "See Also" sections:
```markdown
## Related Documentation
- [related-topic.md](related-topic.md) - Context about relationship
- [another-topic.md](another-topic.md) - Context about relationship
```

## Autonomy Guidelines

**Work autonomously on**:
- File naming decisions
- Content organization
- README section placement
- Commit timing and messages
- Creating cross-references

**Ask for clarification when**:
- Content format is unclear or corrupted
- Multiple valid organizational approaches exist
- Significant overlap with existing files requires strategic decision
- User intent is ambiguous

## Quality Checklist

Before completing integration, verify:
- [ ] All files use kebab-case naming
- [ ] README updated with descriptive entries
- [ ] Files have proper markdown structure
- [ ] Code blocks use appropriate syntax highlighting
- [ ] Cross-references added where relevant
- [ ] All changes committed with descriptive messages
- [ ] All commits pushed to remote
- [ ] Working tree is clean

## Example Session Flow

```
User: [Provides conversation or artifact content]

Claude:
1. Analyzes content and identifies 3 main topics
2. Creates 3 markdown files with descriptive names
3. Updates README with new entries in appropriate section
4. Commits: "Add [topic] documentation"
5. Pushes to remote
6. Reports completion with file summary
```

## Common Patterns

**Multi-phase implementations**:
- Create one file per phase: `project-phase1-foundation.md`, `project-phase2-advanced.md`
- Create specification file separately: `project-specification.md`
- Group under same README subsection

**Comparison topics**:
- Single file comparing approaches: `flycheck-vs-flymake-comparison.md`
- Include recommendation/conclusion section

**Technical guides**:
- Lead with tool name: `ollama-gpu-troubleshooting-guide.md`
- Include complete working examples

**Integration/Setup**:
- Mention both technologies: `claude-code-emacs-configuration.md`
- Include configuration code ready to use

## Error Handling

**API errors during file creation**:
- Retry once immediately
- If fails again, report to user

**Git push permission denied**:
- Verify branch name matches pattern (starts with `claude/`)
- Check session ID matches current session
- Report to user if unable to resolve

**Merge conflicts**:
- Pull latest changes: `git pull origin {branch-name}`
- Report conflict to user for resolution

---

## Start Integration

After receiving this prompt, wait for the user to provide content to integrate. Then follow the workflow above to create well-organized, deduplicated topic files.
