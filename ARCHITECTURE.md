# okf-nav Architecture

## Overview

okf-nav is a single-package Python CLI for navigating, auditing, and exporting [OKF](https://okf.md/) knowledge bundles.

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  CLI (click)  │────▶│  Core Logic  │────▶│  File System  │
│  cli.py       │     │  (bundle     │     │  (OKF bundles │
│  __main__.py  │     │   parsing,   │     │   *.md + YAML │
│               │     │   search,    │     │   frontmatter)│
│               │     │   audit)     │     │               │
└──────────────┘     └─────────────┘     └──────────────┘
                            │
                     ┌──────┴──────┐
                     │  TF-IDF     │
                     │  (optional) │
                     │  scikit-    │
                     │  learn      │
                     └─────────────┘
```

## Package Structure

```
src/okf_nav/
├── __init__.py       # Package metadata, version
├── __main__.py       # python -m okf-nav entry point
└── cli.py            # All CLI commands (click-based)
```

## Commands

| Command | Description |
|---|---|
| `search` | Full-text + optional TF-IDF semantic search |
| `topics` | List all bundles |
| `status` | Bundle overview (counts, health) |
| `show` | Display a specific concept |
| `health` | Check bundle quality |
| `audit` | Check + auto-fix issues |
| `export` | Dump as JSON |
| `update` | Set frontmatter fields / manage tags |
| `stale` | List/archive stale entries (90d+) |
| `context` | Agent context injection (compact pointers) |

## Key Design Decisions

1. **Single-file CLI**: All commands in `cli.py` for simplicity. Split if it grows beyond 1000 lines.
2. **Click framework**: Standard Python CLI library, good for subcommands.
3. **Optional semantic search**: TF-IDF via scikit-learn, behind `[semantic]` extra. No hard dependency.
4. **File-system native**: No database. Bundles are just directories of Markdown + YAML.
5. **Auto-fix strategy**: Audit can auto-repair missing descriptions (extract from body), missing tags (infer from path/type), and stale entries (archive).

## Dependencies

- **Required**: click, pyyaml, rich (terminal formatting)
- **Optional**: scikit-learn (semantic search)
- **Dev**: ruff, pytest, uv

## Publishing

- PyPI via GitHub Actions on release
- Uses `uv build` + `uv publish`
- Version check prevents duplicate uploads
