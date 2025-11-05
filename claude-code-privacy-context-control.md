# Claude Code Privacy and Context Control

Understanding what context gets sent to Claude and how to control it.

## What claude-code-mode Does vs Doesn't Send

**Important**: `claude-code-mode` itself does NOT automatically send context to Claude. It only sets up the infrastructure to enable context sharing when you actively use Claude Code commands.

## Context Sending Scenarios

### Automatic Context (with Monet)

If you're using Monet integration:
- Current selection is automatically shared when you select text
- Diagnostics (errors/warnings) from the current file are sent
- File metadata (name, cursor position) when you switch files

### Manual Context (Standard claude-code.el)

Without Monet, context is only sent when you explicitly:
- Run `claude-code-send-region` - sends selected text or entire buffer
- Run `claude-code-send-command-with-context` - sends filename and line number
- Run `claude-code-send-buffer-file` - sends the entire file
- Run `claude-code-fix-error-at-point` - sends error details and surrounding code

## Context Scope Limitations

Important safeguards built into the system:
- **Project Boundaries**: Claude Code uses Emacs built-in `project.el` which works with most version control systems - Claude only has access to files within the detected project
- **Explicit Commands**: Most context sharing requires deliberate action on your part
- **No Background Monitoring**: The mode doesn't continuously stream your keystrokes or file contents

## Controlling Context Exposure

### Minimal Context Setup
```elisp
;; Enable the mode but don't use Monet
(claude-code-mode 1)
;; This gives you commands but no automatic context sharing
```

### Selective Context Setup
```elisp
;; Use buffer size limits to prevent accidentally sending large files
(setq claude-code-large-buffer-threshold 50000)  ; Confirm before sending files >50KB

;; Disable automatic context in commands
;; Always use claude-code-send-command instead of claude-code-send-command-with-context
```

### Maximum Privacy Setup
```elisp
;; Don't enable global mode - only enable per-project
(add-hook 'some-specific-project-hook 'claude-code-mode)

;; Don't use Monet integration
;; Rely only on explicit copy/paste workflows
```

## Data Flow Summary

| Scenario | What Gets Sent | When |
|----------|---------------|------|
| Mode enabled only | Nothing | Never (just enables commands) |
| With Monet | Selection, diagnostics, file metadata | Automatically when you select text or change files |
| Manual commands | Only what you explicitly send | When you run send commands |
| Large files | Prompts for confirmation | When buffer exceeds threshold |

## Recommendations by Use Case

### High Privacy/Sensitive Code
- Enable mode but skip Monet
- Use small thresholds for buffer size warnings
- Rely on manual `claude-code-send-command` without context

### Balanced Approach
- Use the standard setup with reasonable buffer size limits
- Review what you're sending with `claude-code-send-region` before sending

### Maximum Productivity
- Full Monet integration for seamless context sharing
- Higher buffer size thresholds
- Trust the project boundary protections

## Monet Activation Strategies

### Global Monet with Selective Claude Code
```elisp
;; Enable Monet globally (for the IDE protocol infrastructure)
(monet-mode 1)

;; But control Claude Code mode per-project
(defun my-enable-claude-for-project ()
  (when (my-project-approved-p)
    (claude-code-mode 1)))
```

**Pros**: Monet always ready, but context only shared in approved projects

### Coupled Activation (Higher Security)
```elisp
;; Enable both together only in approved projects
(defun my-enable-claude-with-monet ()
  (when (my-project-approved-p)
    (claude-code-mode 1)
    (monet-mode 1)))
```

**Pros**: Tighter security boundary, no network servers in unapproved projects
**Cons**: Startup delay when first activating in a project

### Monet Behavior

When Monet is enabled but Claude Code mode is not:
- WebSocket server is running (localhost only)
- No automatic context sharing occurs
- Claude Code commands are not available
- IDE protocol infrastructure is ready but inactive

## Key Insight

The actual context sharing depends on which features you use and how you configure them. `claude-code-mode` is just the enabling infrastructure - you maintain full control over what gets sent to Claude through your choice of commands and Monet activation strategy.
