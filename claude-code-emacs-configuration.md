# Claude Code Emacs Configuration

This configuration sets up Claude Code integration in Emacs using the `stevemolitor/claude-code.el` package. Add these code blocks to your literate programming Org mode document.

## Package Installation and Basic Setup

```elisp
;; Install terminal backend (choose one)
;; For vterm (recommended for better performance):
(use-package vterm
  :straight t)

;; For eat (alternative terminal backend):
;; (use-package eat
;;   :straight (:type git
;;              :host codeberg
;;              :repo "akib/emacs-eat"
;;              :files ("*.el" ("term" "term/*.el") "*.texi"
;;                      "*.ti" ("terminfo/e" "terminfo/e/*")
;;                      ("terminfo/65" "terminfo/65/*")
;;                      ("integration" "integration/*")
;;                      (:exclude ".dir-locals.el" "*-tests.el"))))

;; Install and configure claude-code using straight.el
(use-package claude-code
  :straight (:type git
             :host github
             :repo "stevemolitor/claude-code.el"
             :branch "main"
             :depth 1
             :files ("*.el" (:exclude "images/*")))
  :bind-keymap
  ("C-c c" . claude-code-command-map)  ; Set your preferred prefix key
  :config
  ;; Start Emacs server for hook integration
  (unless (server-running-p)
    (start-server))

  ;; Project-specific activation of both Claude Code and Monet
  (defun my-enable-claude-for-approved-projects ()
    "Enable Claude Code mode and Monet for approved projects only."
    (when (and (project-current)
               ;; Add your approved project names here
               (member (project-name (project-current))
                       '("my-open-source-project"
                         "personal-coding-projects"
                         "learning-experiments")))
      (claude-code-mode 1)
      ;; Enable Monet when Claude Code is enabled
      (when (fboundp 'monet-mode)
        (monet-mode 1))))

  ;; Enable both Claude Code and Monet when opening files in approved projects
  (add-hook 'find-file-hook 'my-enable-claude-for-approved-projects))
```

## Flycheck Configuration

```elisp
;; Flycheck for comprehensive syntax checking and linting
(use-package flycheck
  :straight t
  :config
  ;; Enable flycheck globally for programming modes
  (add-hook 'after-init-hook #'global-flycheck-mode)

  ;; Better error display for learning
  (setq flycheck-display-errors-delay 0.3)          ; Show errors quickly
  (setq flycheck-idle-change-delay 0.5)             ; Check after short pause
  (setq flycheck-indication-mode 'right-fringe)     ; Show indicators in right fringe
  (setq flycheck-highlighting-mode 'lines)          ; Highlight entire problematic lines

  ;; Enable error navigation with standard keys
  (define-key flycheck-mode-map (kbd "C-c ! n") 'flycheck-next-error)
  (define-key flycheck-mode-map (kbd "C-c ! p") 'flycheck-previous-error)
  (define-key flycheck-mode-map (kbd "C-c ! l") 'flycheck-list-errors)

  ;; Chain multiple checkers for richer feedback (examples for common languages)
  ;; Python: Use both flake8 and pylint
  (flycheck-add-next-checker 'python-flake8 'python-pylint 'append)

  ;; JavaScript: Use ESLint then JSHint for comprehensive checking
  (flycheck-add-next-checker 'javascript-eslint 'javascript-jshint 'append)

  ;; Enable checkers for common languages (will only activate if tools are installed)
  (setq flycheck-python-flake8-executable "flake8")
  (setq flycheck-python-pylint-executable "pylint"))

;; Emacs Lisp linting support
(use-package package-lint
  :straight t
  :after flycheck
  :config
  ;; Enable package-lint checker for Emacs Lisp files
  ;; Use a safer approach to ensure flycheck-package-lint is loaded
  (when (fboundp 'flycheck-package-lint-setup)
    (flycheck-package-lint-setup))

  ;; Alternative approach - add the checker manually if setup function fails
  (unless (fboundp 'flycheck-package-lint-setup)
    (message "flycheck-package-lint-setup not found, trying alternative setup")
    (when (and (featurep 'flycheck) (featurep 'package-lint))
      (flycheck-define-checker emacs-lisp-package-lint
        "An Emacs Lisp package checker using package-lint."
        :command ("emacs" "-Q" "--batch"
                  "--eval" "(require 'package-lint)"
                  "--eval" "(setq package-lint-batch-fail-on-warnings t)"
                  "-f" "package-lint-batch-and-exit"
                  source-original)
        :error-patterns
        ((warning line-start (file-name) ":" line ":" column ": warning: " (message) line-end)
         (error line-start (file-name) ":" line ":" column ": error: " (message) line-end))
        :modes emacs-lisp-mode)
      (add-to-list 'flycheck-checkers 'emacs-lisp-package-lint 'append))))

;; Additional Emacs Lisp linting (optional but comprehensive)
(use-package elisp-lint
  :straight t
  :after flycheck)

;; Enhanced error display with tooltips (great for learning)
(use-package flycheck-pos-tip
  :straight t
  :after flycheck
  :config
  ;; Show detailed error messages in tooltips on hover
  (flycheck-pos-tip-mode 1)
  ;; Customize tooltip appearance
  (setq flycheck-pos-tip-timeout 30))            ; Keep tooltips visible longer

;; Alternative: Use popup instead of pos-tip (choose one)
;; (use-package flycheck-popup-tip
;;   :straight t
;;   :after flycheck
;;   :config (flycheck-popup-tip-mode))

;; Optional: Inline error display (shows errors right in the buffer)
(use-package flycheck-inline
  :straight t
  :after flycheck
  :config
  ;; Enable inline error display for immediate feedback
  (add-hook 'flycheck-mode-hook #'flycheck-inline-mode))
```

## Terminal Backend Configuration

```elisp
;; Configure terminal backend (default is eat, but vterm is recommended)
(setq claude-code-terminal-backend 'vterm)

;; Terminal-specific configurations
(when (eq claude-code-terminal-backend 'vterm)
  ;; Increase vterm scrollback for long Claude conversations
  (setq vterm-max-scrollback 100000)

  ;; Allow narrow vterm windows (useful for side windows)
  (setopt vterm-min-window-width 40))
```

## Key Binding Customization

```elisp
;; Optional: Define a repeat map for mode cycling
(use-package claude-code
  :bind
  (:repeat-map my-claude-code-map
   ("M" . claude-code-cycle-mode)))

;; Alternative key binding if you prefer different prefix
;; (global-set-key (kbd "C-c C-a") claude-code-command-map)
```

## Display and Window Configuration

```elisp
;; Configure Claude window to appear in a side window
(add-to-list 'display-buffer-alist
             '("^\\*claude"
               (display-buffer-in-side-window)
               (side . right)           ; right, left, top, or bottom
               (window-width . 90)))    ; adjust width as needed

;; Prevent Claude window from being closed by delete-other-windows
(setq claude-code-no-delete-other-windows t)
```

## Notification Configuration

```elisp
;; Enable notifications when Claude finishes processing
(setq claude-code-enable-notifications t)

;; macOS native notifications with sound (optional)
(when (eq system-type 'darwin)
  (defun my-claude-notify (title message)
    "Display a macOS notification with sound."
    (call-process "osascript" nil nil nil
                  "-e" (format "display notification \"%s\" with title \"%s\" sound name \"Glass\""
                               message title)))
  (setq claude-code-notification-function #'my-claude-notify))

;; Linux notifications (optional)
(when (eq system-type 'gnu/linux)
  (defun my-claude-notify (title message)
    "Display a Linux notification using notify-send."
    (if (executable-find "notify-send")
        (call-process "notify-send" nil nil nil title message)
      (message "%s: %s" title message)))
  (setq claude-code-notification-function #'my-claude-notify))
```

## Advanced Configuration

```elisp
;; Customize newline behavior in Claude buffers
(setq claude-code-newline-keybinding-style 'newline-on-shift-return)

;; Enable auto-revert mode for files modified by Claude
(global-auto-revert-mode 1)

;; Increase buffer size threshold for confirmation prompts
(setq claude-code-large-buffer-threshold 150000)

;; Reduce flickering in Claude buffers
(add-hook 'claude-code-start-hook
          (lambda ()
            (when (eq claude-code-terminal-backend 'eat)
              (setq-local eat-minimum-latency 0.033
                          eat-maximum-latency 0.1))))
```

## Font Configuration for Unicode Support

```elisp
;; Configure fonts for proper Unicode character display
;; This is important for Claude's special characters

;; macOS font fallback configuration
(when (eq system-type 'darwin)
  (setq use-default-font-for-symbols nil)
  (set-fontset-font t 'symbol "STIX Two Math" nil 'prepend)
  (set-fontset-font t 'symbol "Menlo" nil 'prepend)
  ;; Add your preferred font last
  (set-fontset-font t 'symbol "SF Mono" nil 'prepend))

;; Cross-platform solution using JuliaMono (requires installing JuliaMono font)
;; (setq use-default-font-for-symbols nil)
;; (set-fontset-font t 'unicode (font-spec :family "JuliaMono"))
```

## Optional IDE Integration with Monet

```elisp
;; Enhanced IDE integration - will be enabled per-project along with Claude Code
(use-package monet
  :straight (:type git :host github :repo "stevemolitor/monet")
  :config
  ;; Set up the integration hook for claude-code.el
  (add-hook 'claude-code-process-environment-functions #'monet-start-server-function)

  ;; Note: monet-mode will be enabled automatically in approved projects
  ;; via the my-enable-claude-for-approved-projects function above

  ;; Optional: Customize Monet behavior
  ;; (setq monet-diff-tool #'monet-ediff-tool)
  ;; (setq monet-diff-cleanup-tool #'monet-ediff-cleanup-tool)
  )
```

## Hook Integration Setup

```elisp
;; Example hook listener for Claude Code CLI events
(defun my-claude-hook-listener (message)
  "Custom listener for Claude Code hooks."
  (let ((hook-type (plist-get message :type))
        (buffer-name (plist-get message :buffer-name))
        (json-data (plist-get message :json-data)))
    (cond
     ((eq hook-type 'notification)
      (message "✅ Claude is ready in %s!" buffer-name))
     ((eq hook-type 'stop)
      (message "🏁 Claude finished in %s!" buffer-name))
     (t
      (message "🤖 Claude hook: %s" hook-type)))))

;; Add the hook listener
(add-hook 'claude-code-event-hook 'my-claude-hook-listener)
```

## Key Commands Reference

The main commands you'll use (with default `C-c c` prefix):

- `C-c c c` - Start Claude
- `C-c c m` - Show transient menu (all commands)
- `C-c c s` - Send command via minibuffer
- `C-c c r` - Send region/buffer to Claude
- `C-c c e` - Fix error at point (works with flycheck/flymake)
- `C-c c t` - Toggle Claude window
- `C-c c b` - Switch to Claude buffer
- `C-c c k` - Kill Claude session
- `C-c c y` - Send "yes" to Claude quickly
- `C-c c n` - Send "no" to Claude quickly

## Claude Code CLI Hook Configuration

To enable CLI hooks, add this to your Claude Code configuration file (`~/.config/claude/config.json`):

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "emacsclient --eval \"(claude-code-handle-hook 'notification \\\"$CLAUDE_BUFFER_NAME\\\")\" \"$(cat)\""
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "emacsclient --eval \"(claude-code-handle-hook 'stop \\\"$CLAUDE_BUFFER_NAME\\\")\" \"$(cat)\""
          }
        ]
      }
    ]
  }
}
```

## Installing Language-Specific Linting Tools

After configuring Flycheck in Emacs, you'll need to install the actual linting tools for the languages you work with:

### Python
```bash
# Install Python linters
pip install flake8 pylint mypy black isort

# Optional: Install additional checkers
pip install bandit        # Security linting
pip install pydocstyle    # Docstring style checking
```

### JavaScript/TypeScript
```bash
# Install Node.js linters globally
npm install -g eslint jshint prettier

# For TypeScript
npm install -g typescript @typescript-eslint/parser @typescript-eslint/eslint-plugin

# Project-specific (recommended)
npm install --save-dev eslint prettier
```

### Other Languages
```bash
# Emacs Lisp
# Install package-lint for Emacs package development
# This checks for common package.el issues and conventions
# Install via Emacs package system or add to your config:
# M-x package-install RET package-lint RET

# For more comprehensive Emacs Lisp checking, also consider:
# M-x package-install RET elisp-lint RET

# Rust (if using Rust)
rustup component add clippy rustfmt

# Go (if using Go)
go install honnef.co/go/tools/cmd/staticcheck@latest
go install golang.org/x/tools/cmd/goimports@latest

# Shell scripts
# Install shellcheck via your system package manager
# Ubuntu/Debian: apt install shellcheck
# macOS: brew install shellcheck
```

### Checking Flycheck Setup

After installation, verify your setup:

1. **Check available checkers**: `M-x flycheck-verify-setup`
2. **List all checkers**: `M-x flycheck-describe-checker`
3. **Toggle flycheck**: `M-x flycheck-mode`
4. **Navigate errors**: `C-c ! n` (next error), `C-c ! p` (previous error)
5. **List errors**: `C-c ! l`
