# Flycheck vs Flymake Comparison

A comprehensive comparison of the two major syntax checking options for Emacs.

## Open Source Status

### Flycheck
✅ Fully open source (GPL-3.0 license)
Repository: https://github.com/flycheck/flycheck
Community-driven project with many contributors
Extensive ecosystem of community-maintained checkers

### Flymake
✅ Fully open source (GPL-3.0+ license)
Built into Emacs core (part of GNU Emacs)
Maintained by the Emacs development team
Ships with every Emacs installation

## Costs

Both are **completely free**:
- ❌ No licensing costs
- ❌ No subscription fees
- ❌ No usage limits
- ❌ No premium features behind paywalls

The only "costs" are:
- Time to configure and learn
- Computational resources (minimal CPU/memory usage)
- Potential costs from external tools they integrate with (but not from Flycheck/Flymake themselves)

## Privacy and External Communication

Neither Flycheck nor Flymake send code to external servers by default.

### What They Actually Do
- **Flycheck**: Orchestrates local linters/checkers (eslint, pylint, rustc, etc.)
- **Flymake**: Same concept - runs local analysis tools and displays results

Both tools work by:
1. Running local analysis tools on your machine
2. Parsing the output from those tools
3. Displaying results in Emacs

### When External Communication Might Occur

External server communication only occurs if:
1. You configure external language servers (LSP)
2. You use cloud-based linters (rare)
3. Language servers that phone home (uncommon - most LSP servers run locally)

This is the LSP server's behavior, not Flycheck/Flymake's.

## Feature Comparison

### Flycheck Advantages
- **More checker support**: 50+ built-in syntax checkers
- **Better error display**: More polished UI with tooltips, margins
- **Active ecosystem**: Lots of community extensions
- **Better documentation**: Comprehensive manual and guides
- **More configuration options**: Highly customizable

### Flymake Advantages
- **Built-in to Emacs**: No installation required
- **Lighter weight**: Simpler codebase, faster startup
- **Better LSP integration**: Improved significantly in modern Emacs
- **Official support**: Maintained by Emacs team
- **Fewer dependencies**: Self-contained

## For Learners vs Experienced Developers

### Flycheck is Better for Learners

**Superior Error Messages and Display**:
- Clearer visual feedback with better highlighting, tooltips, and margin indicators
- More detailed explanations showing not just what's wrong, but hints about why
- Better error categorization with clear distinction between errors, warnings, and info messages
- Tooltips on hover to see full error messages without switching buffers

**More Comprehensive Language Support**:
- 50+ built-in checkers vs Flymake's more limited set
- Multiple checkers per language for richer feedback
- Better beginner-friendly linters focused on style and best practices

**Educational Value**:
- Explanatory checker names so you can see which tool flagged what (pylint vs flake8 vs mypy)
- Gradual complexity - can enable more advanced checkers as you progress
- Style guidance through linters focused on code style and conventions

**Learning-Friendly Features**:
```elisp
;; Make errors more visible for learning
(use-package flycheck-pos-tip
  :after flycheck
  :config (flycheck-pos-tip-mode))

;; Enable multiple checkers for richer feedback
(flycheck-add-next-checker 'python-flake8 'python-pylint)
```

### Flymake is Better for Experienced Developers Who:
- Know their language well and just need basic error detection
- Use primarily LSP servers (which integrate well with modern Flymake)
- Prefer minimal, built-in solutions
- Want faster startup times

## Claude Code Integration

For `claude-code-fix-error-at-point` (`C-c c e`), Flycheck provides better context:
- Detailed error messages from multiple linters
- Style suggestions alongside functional errors
- Clear error categorization (error vs warning vs info)
- More learning opportunities in the suggestions

## Recommendation

### Choose Flycheck if:
- You want maximum language support out of the box
- You prefer polished UI and extensive customization
- You don't mind installing an additional package
- You want community-driven development
- **You are learning to code** (recommended)

### Choose Flymake if:
- You prefer built-in Emacs tools
- You want minimal configuration
- You primarily use LSP servers (which work great with modern Flymake)
- You value official Emacs integration
- You're an experienced developer who needs basic checking

## Security and Privacy

Both are privacy-friendly:
- Code stays on your machine
- No telemetry or analytics
- No network connections (unless you configure external tools)
- Source code is auditable

**Trust Model**:
```
Your Code → Local Linter Tool → Flycheck/Flymake → Display in Emacs
     ↑              ↑                ↑              ↑
   Private      Private          Private        Private
```

## Bottom Line

For Claude Code integration specifically, either works perfectly. The choice comes down to your general Emacs philosophy and experience level rather than Claude Code compatibility. **For learners, Flycheck's richer feedback will accelerate the learning process significantly.**
