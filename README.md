# okf-nav 🧭

**OKF Knowledge Navigator** — search, audit, fix, and export [Open Knowledge Format](https://okf.md/) bundles from your terminal.

```bash
uv tool install okf-nav        # or: pip install okf-nav
okf-nav search "agent memory"   # full-text search
okf-nav audit --fix             # detect + auto-repair
okf-nav export --json           # dump as JSON
```

---

## Why

[OKF](https://okf.md/) bundles are directories of Markdown files with YAML frontmatter — simple, git-friendly, agent-ready. But once you have a bundle, you need tooling to:

- **Search** across concepts (full-text + semantic TF-IDF)
- **Audit** quality (missing descriptions, tags, stale entries)
- **Fix** issues automatically
- **Export** to JSON for pipelines and integrations

okflint validates. Kiso publishes. **okf-nav helps you navigate and maintain.**

---

## Install

```bash
# Recommended — uv (fast, isolated)
uv tool install okf-nav

# Or pip
pip install okf-nav

# With semantic search (TF-IDF)
uv tool install 'okf-nav[semantic]'
```

**Requires:** Python 3.10+

---

## Usage

### Search

```bash
# Full-text search
okf-nav search "agent memory retrieval"

# TF-IDF semantic search (requires scikit-learn)
okf-nav search --semantic "related concepts"

# Show all results including duplicates
okf-nav search --all "database"
```

### Explore

```bash
# List all bundles
okf-nav topics

# Bundle overview
okf-nav status

# Show a specific concept
okf-nav show my-bundle/concepts/foo.md
```

### Health & Fix

```bash
# Check bundle quality
okf-nav health

# Check + auto-fix missing descriptions and tags
okf-nav audit --fix
```

Auto-fix repairs:
- **Missing description** → extracts from body text
- **Missing tags** → infers from type and path
- **Stale entries** (≥90d) → archives to `.archive/`

### Export

```bash
# All bundles as JSON
okf-nav export --json

# Single bundle, minified
okf-nav export --json --bundle my-bundle --minify
```

### Edit

```bash
# Set frontmatter fields
okf-nav update my-bundle/concepts/foo.md --set description="New desc"

# Manage tags
okf-nav update foo.md --add-tag python --remove-tag legacy
```

### Stale

```bash
# List entries not updated in 90+ days
okf-nav stale

# Archive stale entries
okf-nav stale --archive

# Custom threshold
okf-nav stale --older 30
```

### Agent Integration

```bash
# Auto-context hook: returns relevant pointers for a prompt
okf-nav context "How do I configure the memory system?"
```

This outputs a compact pointer block with inline snippet — designed for agent context injection (`<!-- okf-nav:context -->`).

---

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `OKF_BUNDLES_DIR` | `./okf-bundles` | Directory containing OKF bundles |
| `OKF_CACHE_DIR` | `~/.cache/okf-nav` | Cache location for search indices |

Point to your bundle:

```bash
OKF_BUNDLES_DIR=/path/to/your/bundles okf-nav status
```

---

## Comparison

| Feature | okf-nav | okflint | superops/okf | Kiso |
|---------|---------|---------|--------------|------|
| Search (full-text) | ✅ | ❌ | ✅ | ❌ |
| Search (semantic) | ✅ | ❌ | ❌ | ❌ |
| Validate | ✅ basic | ✅ deep (18 rules) | ✅ 10 rules | ✅ check |
| Auto-fix | ✅ | ❌ | ❌ | ❌ |
| JSON export | ✅ | ✅ (diagnostics) | ❌ | ❌ |
| Static site | ❌ | ❌ | ❌ | ✅ |
| Git integration | ❌ | ❌ | ✅ | ❌ |
| CI gate | ❌ | ✅ | ❌ | ❌ |

**okf-nav fills the gap**: okflint validates, okf-nav fixes + searches + exports.

---

## Project Status

Alpha. Active development. The CLI is functional and used daily by the author.

---

## License

MIT
