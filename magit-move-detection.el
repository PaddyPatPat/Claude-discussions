;;; magit-move-detection.el --- Detect content moves between org-roam files in Magit -*- lexical-binding: t; -*-

;; Copyright (C) 2025

;; Author: Claude (AI Assistant)
;; Version: 2.5
;; Package-Requires: ((emacs "26.1") (magit "3.0"))
;; Keywords: vc, tools, org-mode
;; URL: https://github.com/yourusername/magit-move-detection

;;; Commentary:

;; This package detects when content has been moved between org-roam files
;; and displays these moves in the Magit status buffer.  It helps streamline
;; the git commit workflow when refactoring Zettelkasten notes.
;;
;; Key features:
;; - Detects content moves with >85% similarity
;; - Groups multi-line hunks automatically
;; - Handles org-mode heading level changes
;; - Prioritizes largest moves first
;; - Uses Levenshtein distance for robust matching
;;
;; Usage:
;;   1. Make changes involving content moves between org files
;;   2. Open magit-status (M-x magit-status)
;;   3. Run M-x org-roam-move-debug
;;   4. Review detected moves in *Org-Roam Move Debug* buffer
;;
;; Future versions will integrate directly into magit-status display.

;;; Code:

(require 'magit)
(require 'seq)

;;; Configuration

(defconst org-roam-move-debug-version "2.5"
  "Version of the org-roam move detection prototype.")

(defvar org-roam-move-similarity-threshold 85
  "Minimum similarity percentage for detecting moves (0-100).
Higher values require more exact matches.  Default of 85 allows
for heading level changes and minor edits.")

(defvar org-roam-move-minimum-hunk-size 0
  "Minimum character count for hunks to be considered.
Set to 0 to consider all hunks.  Increase to filter out very
small matches that are likely coincidental.")

(defvar org-roam-move-debug-similarity-threshold 30
  "Minimum similarity to show in debug output.
Set lower than `org-roam-move-similarity-threshold' to see
near-miss comparisons during debugging.")

;;; Core Functions

;;;###autoload
(defun org-roam-move-debug ()
  "Debug function to explore magit section structure and find potential moves.
Run this in a magit-status buffer to see what hunk data we can access.
Displays results in *Org-Roam Move Debug* buffer."
  (interactive)
  (unless (derived-mode-p 'magit-status-mode)
    (error "Must be run in a magit-status buffer"))

  (let ((deleted-hunks '())
        (added-hunks '())
        (debug-info '())
        (all-comparisons '())
        (current-file nil)
        (current-group '())
        (current-prefix nil))

    ;; Parse buffer and extract hunks
    (save-excursion
      (goto-char (point-min))
      (let ((line-num 0))
        (while (not (eobp))
          (let ((line (buffer-substring-no-properties
                      (line-beginning-position)
                      (line-end-position))))

            ;; Update file context
            (when (string-match "^modified[ \t]+\\([a-zA-Z0-9_/-]+\\.org\\)" line)
              (setq current-file (match-string 1 line))
              (push (format "Line %d: Set current-file to %s" line-num current-file)
                    debug-info))

            ;; Handle diff lines
            (when (string-match "^\\([-+]\\)" line)
              (let* ((prefix (match-string 1 line))
                     (content (substring line 1))) ; Remove prefix

                (cond
                 ;; Start new deletion group
                 ((and (string= prefix "-")
                       (not (string= current-prefix "-")))
                  (when current-group
                    (org-roam-finalize-hunk current-group current-file current-prefix
                                           deleted-hunks added-hunks debug-info line-num))
                  (setq current-group (list content)
                        current-prefix "-")
                  (push (format "Line %d: Started deletion group in %s"
                               line-num current-file)
                        debug-info))

                 ;; Continue deletion group
                 ((string= prefix "-")
                  (push content current-group)
                  (push (format "Line %d: Added to deleted group in %s: '%s'"
                               line-num current-file content)
                        debug-info))

                 ;; Start new addition group
                 ((and (string= prefix "+")
                       (not (string= current-prefix "+")))
                  (when current-group
                    (org-roam-finalize-hunk current-group current-file current-prefix
                                           deleted-hunks added-hunks debug-info line-num))
                  (setq current-group (list content)
                        current-prefix "+")
                  (push (format "Line %d: Started addition group in %s"
                               line-num current-file)
                        debug-info))

                 ;; Continue addition group
                 ((string= prefix "+")
                  (push content current-group)
                  (push (format "Line %d: Added to added group in %s: '%s'"
                               line-num current-file content)
                        debug-info)))))

            ;; Context line ends current group
            (when (and (not (string-match "^[-+]" line))
                      (not (string-match "^@@" line))
                      (not (string-match "^modified" line))
                      current-group)
              (org-roam-finalize-hunk current-group current-file current-prefix
                                     deleted-hunks added-hunks debug-info line-num)
              (setq current-group nil
                    current-prefix nil))

            (setq line-num (1+ line-num)))
          (forward-line 1)))

      ;; Finalize any remaining group
      (when current-group
        (org-roam-finalize-hunk current-group current-file current-prefix
                               deleted-hunks added-hunks debug-info -1)))

    ;; Sort hunks by size (largest first)
    (setq deleted-hunks (sort deleted-hunks
                             (lambda (a b) (> (plist-get a :size)
                                             (plist-get b :size)))))
    (setq added-hunks (sort added-hunks
                           (lambda (a b) (> (plist-get a :size)
                                           (plist-get b :size)))))

    ;; Find potential moves
    (let ((potential-moves '()))
      (dolist (deleted deleted-hunks)
        (dolist (added added-hunks)
          ;; Only compare across different files
          (when (not (string= (plist-get deleted :file)
                             (plist-get added :file)))
            (let* ((del-content (org-roam-normalize-content
                                (plist-get deleted :content)))
                   (add-content (org-roam-normalize-content
                                (plist-get added :content)))
                   (similarity (org-roam-calculate-similarity
                               del-content add-content)))

              ;; Track all comparisons for debugging
              (when (> similarity org-roam-move-debug-similarity-threshold)
                (push (list :from-file (plist-get deleted :file)
                           :to-file (plist-get added :file)
                           :similarity similarity
                           :from-content (truncate-string-to-width del-content 30)
                           :to-content (truncate-string-to-width add-content 30))
                      all-comparisons)
                (push (format "MATCH FOUND: %s->%s (%.1f%%) '%s' vs '%s'"
                             (plist-get deleted :file)
                             (plist-get added :file)
                             similarity
                             (truncate-string-to-width del-content 30)
                             (truncate-string-to-width add-content 30))
                      debug-info))

              ;; High-confidence moves only
              (when (> similarity org-roam-move-similarity-threshold)
                (push (list :from-file (plist-get deleted :file)
                           :to-file (plist-get added :file)
                           :content del-content
                           :similarity similarity
                           :size (plist-get deleted :size))
                      potential-moves))))))

      ;; Sort moves by size (largest first)
      (setq potential-moves (sort potential-moves
                                 (lambda (a b) (> (plist-get a :size)
                                                 (plist-get b :size)))))

      ;; Display results
      (org-roam-display-results deleted-hunks added-hunks potential-moves
                               all-comparisons debug-info))))

(defun org-roam-finalize-hunk (group file prefix deleted-hunks added-hunks debug-info line-num)
  "Finalize a grouped hunk and add to appropriate list.
GROUP is list of content lines, FILE is the filename, PREFIX is '-' or '+'.
Adds to DELETED-HUNKS or ADDED-HUNKS.  Updates DEBUG-INFO with line LINE-NUM."
  (let ((hunk (org-roam-create-hunk-from-group group file
                                              (if (string= prefix "-")
                                                  'deleted
                                                'added))))
    (when hunk
      (if (string= prefix "-")
          (push hunk deleted-hunks)
        (push hunk added-hunks))
      (push (format "Line %d: Finalized %s hunk in %s with %d lines"
                   line-num
                   (if (string= prefix "-") "deleted" "added")
                   file
                   (plist-get hunk :lines))
            debug-info))))

(defun org-roam-create-hunk-from-group (lines file type)
  "Create hunk object from grouped LINES for FILE with TYPE.
Filters empty lines and returns property list with :file, :content, :type, :size, :lines."
  (let ((filtered-lines (seq-filter
                         (lambda (line)
                           (not (string-match-p "^[ \t]*$" line)))
                         lines)))
    (when filtered-lines
      (let* ((content (mapconcat #'identity
                                (reverse filtered-lines)
                                "\n"))
             (size (length content))
             (line-count (length filtered-lines)))
        (list :file file
              :content content
              :type type
              :size size
              :lines line-count)))))

(defun org-roam-normalize-content (content)
  "Normalize org-mode CONTENT for comparison.
Removes heading markers, normalizes whitespace, and trims."
  (let ((normalized content))
    ;; Remove org heading asterisks
    (setq normalized (replace-regexp-in-string "^\\*+ " "" normalized))

    ;; Collapse multiple spaces/tabs to single space
    (setq normalized (replace-regexp-in-string "[ \t]+" " " normalized))

    ;; Reduce multiple newlines to single newline
    (setq normalized (replace-regexp-in-string "\n\n+" "\n" normalized))

    ;; Trim leading and trailing whitespace
    (string-trim normalized)))

(defun org-roam-calculate-similarity (str1 str2)
  "Calculate similarity percentage between STR1 and STR2.
Uses Levenshtein distance algorithm.  Returns value 0-100."
  (if (or (string-empty-p str1) (string-empty-p str2))
      0
    (let ((dist (levenshtein-distance str1 str2))
          (maxlen (max (length str1) (length str2))))
      (if (zerop maxlen)
          100
        (* 100.0 (- 1.0 (/ (float dist) maxlen)))))))

(defun levenshtein-distance (str1 str2)
  "Calculate Levenshtein distance between STR1 and STR2.
Returns minimum number of single-character edits (insertions,
deletions, substitutions) needed to transform STR1 into STR2."
  (let* ((len1 (length str1))
         (len2 (length str2))
         (matrix (make-vector (1+ len1) nil)))

    ;; Initialize matrix
    (dotimes (i (1+ len1))
      (aset matrix i (make-vector (1+ len2) 0))
      (aset (aref matrix i) 0 i))
    (dotimes (j (1+ len2))
      (aset (aref matrix 0) j j))

    ;; Fill matrix with dynamic programming
    (dotimes (i len1)
      (dotimes (j len2)
        (let* ((cost (if (= (aref str1 i) (aref str2 j)) 0 1))
               (deletion (1+ (aref (aref matrix i) (1+ j))))
               (insertion (1+ (aref (aref matrix (1+ i)) j)))
               (substitution (+ (aref (aref matrix i) j) cost)))
          (aset (aref matrix (1+ i)) (1+ j)
                (min deletion insertion substitution)))))

    ;; Return final distance
    (aref (aref matrix len1) len2)))

(defun org-roam-display-results (deleted-hunks added-hunks potential-moves all-comparisons debug-info)
  "Display detection results in debug buffer.
Shows DELETED-HUNKS, ADDED-HUNKS, POTENTIAL-MOVES, ALL-COMPARISONS, and DEBUG-INFO."
  (with-current-buffer (get-buffer-create "*Org-Roam Move Debug*")
    (erase-buffer)
    (insert (format "=== ORG-ROAM MOVE DEBUG - VERSION %s ===\n\n"
                    org-roam-move-debug-version))

    ;; Deleted hunks section
    (insert "=== DELETED HUNKS (by size) ===\n\n")
    (if deleted-hunks
        (dolist (hunk (seq-take deleted-hunks 10))
          (insert (format "File: %s | Size: %d | Lines: %d\nContent: '%s'\n\n"
                         (plist-get hunk :file)
                         (plist-get hunk :size)
                         (plist-get hunk :lines)
                         (plist-get hunk :content))))
      (insert "No deleted hunks found.\n"))

    ;; Added hunks section
    (insert "\n=== ADDED HUNKS (by size) ===\n\n")
    (if added-hunks
        (dolist (hunk (seq-take added-hunks 10))
          (insert (format "File: %s | Size: %d | Lines: %d\nContent: '%s'\n\n"
                         (plist-get hunk :file)
                         (plist-get hunk :size)
                         (plist-get hunk :lines)
                         (plist-get hunk :content))))
      (insert "No added hunks found.\n"))

    ;; Potential moves section
    (insert "\n=== POTENTIAL MOVES (by size) ===\n\n")
    (if potential-moves
        (dolist (move potential-moves)
          (insert (format "MOVE: %s → %s\n"
                         (plist-get move :from-file)
                         (plist-get move :to-file)))
          (insert (format "Similarity: %.1f%% | Size: %d\nContent: %s\n\n"
                         (plist-get move :similarity)
                         (plist-get move :size)
                         (plist-get move :content))))
      (insert (format "No potential moves found above %d%% similarity threshold.\n"
                     org-roam-move-similarity-threshold)))

    ;; All comparisons section (for debugging)
    (insert "\n=== ALL SIMILARITY COMPARISONS ===\n")
    (insert (format "(Showing matches above %d%% threshold)\n\n"
                   org-roam-move-debug-similarity-threshold))
    (if all-comparisons
        (dolist (comp (seq-take (sort all-comparisons
                                     (lambda (a b)
                                       (> (plist-get a :similarity)
                                          (plist-get b :similarity))))
                               20))
          (insert (format "%s → %s: %.1f%%\n  '%s' vs '%s'\n"
                         (plist-get comp :from-file)
                         (plist-get comp :to-file)
                         (plist-get comp :similarity)
                         (plist-get comp :from-content)
                         (plist-get comp :to-content))))
      (insert "No comparisons above threshold.\n"))

    ;; Debug log
    (insert "\n=== DEBUG LOG ===\n\n")
    (dolist (info (reverse debug-info))
      (insert info "\n"))

    (goto-char (point-min))
    (pop-to-buffer (current-buffer))))

;;; Future Integration Functions (Placeholders)

;; These functions are placeholders for future Magit integration

(defun magit-insert-org-moves-section ()
  "Insert custom Magit section showing detected moves.
Not yet implemented - currently using debug buffer instead."
  ;; TODO: Implement custom magit section
  ;; (magit-insert-section (org-moves)
  ;;   (magit-insert-heading "Content Moves")
  ;;   ...)
  )

(defun magit-stage-move-pair ()
  "Stage both files involved in move at point.
Not yet implemented."
  (interactive)
  ;; TODO: Implement staging command
  (user-error "Staging commands not yet implemented in v2.5"))

(provide 'magit-move-detection)

;;; magit-move-detection.el ends here
