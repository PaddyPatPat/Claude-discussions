# Claude Code Error Fixing Workflow in Emacs

The `claude-code-fix-error-at-point` command (`C-c c e`) creates a seamless workflow for fixing errors directly from your code.

## What This Command Does

`claude-code-fix-error-at-point` automatically detects errors/warnings at your cursor position and asks Claude to fix them, along with relevant context.

## How Error Detection Works

### Integration with Emacs Linters

The command works with:
- **Flycheck**: Popular third-party syntax checker
- **Flymake**: Built-in Emacs syntax checker
- **LSP servers**: Via flycheck/flymake integration (TypeScript, Python, Rust, etc.)
- **Any help-at-point system**: Custom error reporting systems

### What Gets Detected

- Syntax errors
- Type errors
- Linting warnings
- LSP diagnostics
- Compilation errors
- Static analysis warnings

## The Workflow in Practice

### Example Scenario

```python
# You have this Python code with a type error:
def calculate_total(items: list[str]) -> int:
    return sum(items)  # ← cursor here, red squiggly line
    #      ^^^^^^^^^^
    #      Error: can't sum strings
```

### What Happens When You Press `C-c c e`

1. **Error Detection**: Emacs detects there's a flycheck/flymake error at your cursor
2. **Context Gathering**: The command collects:
   - The exact error message: `"TypeError: unsupported operand type(s) for +: 'int' and 'str'"`
   - The line with the error: `return sum(items)`
   - Surrounding code context (a few lines before/after)
   - File name and line number

3. **Claude Request**: Sends something like:
   ```
   Fix the error at line 2 in calculate_total.py:

   Error: TypeError: unsupported operand type(s) for +: 'int' and 'str'

   Code context:
   def calculate_total(items: list[str]) -> int:
       return sum(items)  # ← Error here
   ```

4. **Claude Response**: Claude analyzes and suggests a fix:
   ```python
   # The issue is that you're trying to sum strings. You probably want:
   def calculate_total(items: list[str]) -> int:
       return sum(int(item) for item in items)
   # Or if items are prices:
   def calculate_total(items: list[str]) -> float:
       return sum(float(item) for item in items)
   ```

## Advanced Error Scenarios

### LSP Integration Example

```typescript
// TypeScript with LSP running
const users: User[] = fetchUsers();
const names = users.map(u => u.fullName);  // ← Property 'fullName' doesn't exist
```

After `C-c c e`:
- Detects TypeScript LSP error
- Sends the interface definition context
- Claude suggests: `u => u.firstName + ' ' + u.lastName` or asks to check the User interface

### Multiple Errors

If there are multiple errors at the cursor position, it gathers all of them:

```javascript
const result = someFunction(undefined, null);  // Multiple argument errors
```

## Context Intelligence

### Smart Context Selection

The command doesn't just send the error line - it intelligently includes:
- Function/method signature
- Relevant imports
- Variable declarations in scope
- Related type definitions (if available)

### File Context

The command supports prefix arguments:
- With prefix arg (`C-u C-c c e`): Switches to the Claude buffer after sending

## Integration with Monet

If you have Monet enabled, this becomes even more powerful:
- Claude can see real-time diagnostics from your linters automatically
- The error context is even richer with project-wide type information
- Claude can see related files and their diagnostics

## Practical Usage Patterns

### Quick Fix Workflow
```
1. See red squiggly
2. C-c c e
3. Get fix suggestion
4. Apply fix
```

### Learning Workflow
```
1. See warning you don't understand
2. C-c c e
3. Get explanation + fix
4. Learn new pattern
```

### Refactoring Workflow
```
1. Make breaking change
2. Multiple errors appear
3. C-c c e on each
4. Get systematic fixes
```

## What It Can Fix Well

- Type errors with clear solutions
- Missing imports
- Simple logic errors
- API usage mistakes
- Syntax errors

## What It Struggles With

- Complex architectural issues
- Performance problems
- Business logic errors
- Ambiguous requirements

## Best Practices

1. **Use descriptive variable names** - helps Claude understand intent
2. **Have good test coverage** - Claude can reference tests for context
3. **Keep functions focused** - easier for Claude to understand scope
4. **Fix errors incrementally** - don't try to fix 20 errors at once

## Configuration Options

```elisp
;; Switch to Claude buffer after sending error (optional)
;; Use C-u C-c c e instead of C-c c e
(setq claude-code-switch-after-error-fix t)

;; Customize how much context to send with errors
(setq claude-code-error-context-lines 10)  ; Lines before/after error
```

## Key Commands for Error Navigation

When used with Flycheck:
- `C-c ! n` - Next error
- `C-c ! p` - Previous error
- `C-c ! l` - List all errors
- `C-c c e` - Fix error at point with Claude

This feature essentially turns Claude into an intelligent pair programmer who can see your linter output and suggest fixes with full context awareness. It's particularly powerful when combined with strong type systems (TypeScript, Rust, etc.) or comprehensive linters.
