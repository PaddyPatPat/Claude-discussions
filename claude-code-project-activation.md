# Claude Code Project-Based Activation in Emacs

How to configure Claude Code and Monet to activate only for approved projects, providing security and resource control.

## Why Project-Based Activation

### Security Benefits
- Control which codebases Claude can access
- Prevent accidental context sharing from sensitive projects
- Clear audit trail of where Claude is enabled

### Resource Benefits
- No overhead in projects where you don't need Claude
- Focused resource usage
- Cleaner separation between work contexts

## How Project Detection Works

Emacs' built-in `project.el` typically detects projects by:
- **Git repositories** - looks for `.git` directory
- **Directory name** - uses the folder name as project name
- **Root detection** - finds the repository root automatically

## Basic Setup

### Finding Your Project Name

To determine what `project.el` calls your project:
1. Open any file in your project directory
2. Run: `M-: (project-name (project-current))`
3. Note the result (likely the directory name)

### Configuring Approved Projects

```elisp
(defun my-enable-claude-for-approved-projects ()
  "Enable Claude Code mode and Monet for approved projects only."
  (when (and (project-current)
             (member (project-name (project-current))
                     '("mac-setup"           ; Your actual project names
                       "my-web-app"
                       "learning-rust")))
    (claude-code-mode 1)
    (when (fboundp 'monet-mode)
      (monet-mode 1))))

(add-hook 'find-file-hook 'my-enable-claude-for-approved-projects)
```

## Alternative Activation Strategies

### Directory-Based Detection

```elisp
(defun my-enable-claude-for-approved-projects ()
  "Enable Claude Code for projects in specific directories."
  (when (and buffer-file-name
             (or (string-match-p "/code/approved/" buffer-file-name)
                 (string-match-p "~/personal-projects/" buffer-file-name)))
    (claude-code-mode 1)
    (when (fboundp 'monet-mode)
      (monet-mode 1))))
```

### Git Repository Based

```elisp
(defun my-enable-claude-for-approved-projects ()
  "Enable Claude Code for specific git repositories."
  (when-let* ((project (project-current))
              (root (project-root project))
              (git-config (expand-file-name ".git/config" root)))
    (when (file-exists-p git-config)
      (with-temp-buffer
        (insert-file-contents git-config)
        (when (or (search-forward "github.com/myusername" nil t)
                  (search-forward "my-company.com" nil t))
          (claude-code-mode 1)
          (when (fboundp 'monet-mode)
            (monet-mode 1)))))))
```

### Interactive Approval

```elisp
(defun my-enable-claude-for-approved-projects ()
  "Ask before enabling Claude Code for new projects."
  (when (and (project-current)
             (not (boundp 'claude-code-project-approved)))
    (when (y-or-n-p "Enable Claude Code for this project? ")
      (claude-code-mode 1)
      (when (fboundp 'monet-mode)
        (monet-mode 1))
      (setq-local claude-code-project-approved t))))
```

## Activation Scope Options

### Coupled Activation (Higher Security)

Both Claude Code and Monet activate together only in approved projects:

```elisp
(defun my-enable-claude-for-approved-projects ()
  "Enable both Claude Code and Monet for approved projects only."
  (when (my-project-approved-p)
    (claude-code-mode 1)
    (when (fboundp 'monet-mode)
      (monet-mode 1))))

(add-hook 'find-file-hook 'my-enable-claude-for-approved-projects)
```

**Pros**:
- Tighter security boundary
- No network servers running in unapproved projects
- Clear separation

**Cons**:
- Startup delay when first activating in a project

### Global Monet with Selective Claude Code

Monet always available, but Claude Code only in approved projects:

```elisp
;; Enable Monet globally
(monet-mode 1)

;; But control Claude Code per-project
(defun my-enable-claude-for-approved-projects ()
  (when (my-project-approved-p)
    (claude-code-mode 1)))

(add-hook 'find-file-hook 'my-enable-claude-for-approved-projects)
```

**Pros**:
- Monet infrastructure always ready
- No startup delay
- Easy to quickly enable Claude in a new project

**Cons**:
- WebSocket server running even in unapproved projects

## Testing Your Setup

After configuring project-based activation:

1. **Test in approved project**:
   - Open a file in an approved project
   - `C-c c c` should start Claude Code
   - Verify both modes active: `M-x describe-mode`

2. **Test in non-approved project**:
   - Open a file in an unapproved project
   - `C-c c c` should not work
   - Verify modes inactive

3. **Check mode status**:
   - Look at mode line for indicators
   - Run `M-x describe-mode` to see active modes

## Troubleshooting

### Project Not Detected

If `project.el` doesn't detect your repository:

```elisp
;; Alternative: Use directory-based detection
(defun my-enable-claude-for-approved-projects ()
  (when (and buffer-file-name
             (string-match-p "/mac-setup/" buffer-file-name))
    (claude-code-mode 1)
    (when (fboundp 'monet-mode)
      (monet-mode 1))))
```

### Modes Not Activating

Check:
- Project name matches exactly: `M-: (project-name (project-current))`
- Hook is registered: `M-x describe-variable find-file-hook`
- No errors in `*Messages*` buffer

## Example: Real-World Configuration

```elisp
(defun my-enable-claude-for-approved-projects ()
  "Enable Claude Code mode and Monet for approved projects only."
  (when (and (project-current)
             ;; List your actual project names here
             (member (project-name (project-current))
                     '("mac-setup"              ; System configuration
                       "my-emacs-config"        ; Emacs setup
                       "learning-python"        ; Learning projects
                       "side-project-app")))    ; Personal projects
    (claude-code-mode 1)
    ;; Enable Monet when Claude Code is enabled
    (when (fboundp 'monet-mode)
      (monet-mode 1))
    ;; Optional: Visual feedback
    (message "✅ Claude Code activated for %s"
             (project-name (project-current)))))

;; Enable on file open
(add-hook 'find-file-hook 'my-enable-claude-for-approved-projects)
```

## Adding New Projects

To add a new project to your approved list:

1. Open a file in the new project
2. Check the project name: `M-: (project-name (project-current))`
3. Add it to your approved projects list
4. Reload configuration: `M-x eval-buffer`
5. Reopen a file from that project to activate

## Security Considerations

- **Sensitive codebases**: Keep proprietary or NDA-protected code out of approved list
- **Personal/experimental**: Safe to include learning and personal projects
- **Client work**: Only include with explicit permission
- **Audit trail**: Your Emacs config serves as documentation of which projects have Claude access
