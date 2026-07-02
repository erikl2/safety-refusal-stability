# Backup

This repo is local-only (no remote, by policy) or has unpushed work. It is backed up
as a git bundle to a private iCloud folder:

    ~/Library/Mobile Documents/com~apple~CloudDocs/code_workspace_backups/bundles/

- **Last bundle:** 2026-07-01 (`safety-refusal-stability_2026-07-01.bundle`, all refs)
- **Refresh:** from the repo root run:
  ```sh
  git bundle create "$HOME/Library/Mobile Documents/com~apple~CloudDocs/code_workspace_backups/bundles/safety-refusal-stability_$(date +%F).bundle" --all
  git bundle verify "$HOME/Library/Mobile Documents/com~apple~CloudDocs/code_workspace_backups/bundles/safety-refusal-stability_$(date +%F).bundle"
  ```
  Delete bundles older than the two most recent.
- **Restore:** `git clone <bundle-file> <dir>`
- Bundles only contain *committed* work — commit before refreshing.

## Sole-copy data (data/results/, 266 MB — gitignored, NOT in the bundle)

Backed up separately as
`code_workspace_backups/data/safety-refusal-stability_data-results_2026-07-01.tar.gz`
(66 MB compressed, 176 entries) with a `SHA256SUMS` manifest verified after copy.
Refresh: re-tar `data/results`, recompute the SHA256 line, re-verify with
`shasum -a 256 -c SHA256SUMS` in the destination dir.
