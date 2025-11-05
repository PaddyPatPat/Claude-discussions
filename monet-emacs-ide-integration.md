# Monet Emacs IDE Integration

Monet is an Emacs package that implements the Claude Code IDE protocol, enabling Claude to interact with your Emacs environment through a WebSocket connection.

**Repository**: https://github.com/stevemolitor/monet

## Primary Benefits

### Real-Time IDE Integration

Monet provides:
- **Automatic Selection Context**: Current selection in Emacs is automatically shared with Claude Code
- **Live Diagnostics**: Send diagnostics from Flymake/Flycheck (and thus LSP in LSP modes) to Claude
- **Interactive Diff Views**: Create diff views in Emacs before Claude applies changes

### Enhanced Workflow Features

- **Intelligent Context Awareness**: Claude automatically knows what code you're looking at and any errors/warnings in your files without manual sharing
- **Advanced Diff Management**: When Claude proposes code changes, Monet displays them in a diff view with options for:
  - Simple Diff Tool (read-only)
  - Ediff Tool (interactive editing)
- **Project-Aware Sessions**: Project-aware session management with multiple concurrent sessions support

### Developer Experience Improvements

- **No Manual File Sharing**: Instead of copying/pasting code snippets, Claude can directly access and understand your project structure
- **Error-Driven Development**: Claude can see linter/LSP errors in real-time and suggest fixes contextually
- **Safe Code Review**: Preview changes before applying them, with the ability to edit Claude's suggestions
- **Multi-Project Support**: Run separate Claude sessions for different projects simultaneously

## Practical Workflow Comparison

### Without Monet
1. Manually copy code
2. Describe errors to Claude
3. Paste Claude's suggestions back into files

### With Monet
1. Claude sees your current selection automatically
2. Claude knows about compilation errors, linter warnings, and LSP diagnostics
3. Claude can browse your project structure
4. You get a proper diff interface to review and refine changes before applying them
5. Changes are applied directly to files through Emacs

## Diff Tool Options

### Simple Diff Tool (Default)
- Shows a read-only side-by-side comparison
- Press `y` to accept Claude's changes exactly as shown
- Press `q` to reject the changes
- Faster workflow for when you trust Claude's suggestions

### Ediff Tool (Advanced)
- Interactive diff view that allows you to edit the changes before accepting
- Navigate between differences using `n` (next) and `p` (previous)
- Edit the proposed changes directly in the buffer
- Press `C-c C-c` to accept your edited version (your changes will be sent to Claude)
- Press `q` to reject all changes

**Configuration**:
```elisp
;; Enable advanced editing capabilities
(setq monet-diff-tool #'monet-ediff-tool)
(setq monet-diff-cleanup-tool #'monet-ediff-cleanup-tool)
```

## When to Use Monet

### Use Monet If You:
- Work on larger codebases where context switching is expensive
- Use LSP servers, flycheck, or flymake for error detection
- Want Claude to understand your project structure automatically
- Prefer reviewing code changes in a proper diff interface
- Work on multiple projects and want isolated Claude sessions

### Skip Monet If You:
- Primarily work with small, single-file projects
- Prefer manual control over what Claude sees
- Don't use linters or LSP servers
- Are comfortable with copy/paste workflows

## Session-Level Feedback Loop

When you make manual edits to Claude's proposed changes in ediff and accept them, your changes are captured and sent to Claude. However, this feedback is **within the current conversation session only** - not persistent learning across sessions.

**What This Enables**:
- Claude can incorporate your style in subsequent changes within the same conversation
- Iterative refinement during a single coding session
- Teaching by demonstration for the current task

**What This Does NOT Do**:
- No persistent learning across different Claude Code sessions
- No global preference storage that affects future projects
- No model fine-tuning based on your edits
- No cross-conversation memory of your coding style

For persistent preferences, document them explicitly in your `claude.md` file.
