#!/usr/bin/env python3
"""okf-nav — OKF Knowledge Navigator.

Search, audit, fix, and export Open Knowledge Format (OKF) bundles.

Usage:
  okf-nav search <query>              Full-text search across all bundles
  okf-nav search --semantic <query>    TF-IDF semantic search
  okf-nav show <path>                  Show a concept document
  okf-nav status                       Bundle overview stats
  okf-nav topics                       List all bundles
  okf-nav health                       Run health check
  okf-nav audit [--fix]                Health check + auto-fix
  okf-nav export --json [--bundle x] [--minify]  Export bundles as JSON
  okf-nav update <path> [--set k=v] [--add-tag t] [--remove-tag t]
  okf-nav stale [--older 90] [--archive]
  okf-nav index                        Rebuild bundle indices
  okf-nav context <prompt>             Auto-context for agents

Environment:
  OKF_BUNDLES_DIR  ./okf-bundles  (default, relative to cwd)
  OKF_CACHE_DIR    ~/.cache/okf-nav
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ── Config ────────────────────────────────────────────────────────
CACHE_DIR = Path(os.environ.get("OKF_CACHE_DIR", Path.home() / ".cache" / "okf-nav"))
BUNDLES_DIR = Path(os.environ.get("OKF_BUNDLES_DIR", "okf-bundles"))

# TF-IDF imports (optional)
_HAS_TFIDF = False
_TFIDF_ERR = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _HAS_TFIDF = True
except ImportError as _e:
    _TFIDF_ERR = str(_e)

# ── Bundle discovery ──────────────────────────────────────────────

def get_bundles() -> dict[str, Path]:
    """Discover OKF bundles under BUNDLES_DIR."""
    bundles: dict[str, Path] = {}
    resolved = BUNDLES_DIR.resolve() if not BUNDLES_DIR.is_absolute() else BUNDLES_DIR
    if resolved.exists():
        for d in sorted(resolved.iterdir()):
            if d.is_dir() and (d / "index.md").exists():
                bundles[d.name] = d
    return bundles


# ── Frontmatter parsing ───────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract YAML frontmatter from markdown content.

    Returns:
        Tuple of (frontmatter_dict, body) or (None, text) if absent.
    """
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    try:
        fm = yaml.safe_load(parts[1])
        if not isinstance(fm, dict):
            return None, text
        return fm, parts[2]
    except yaml.YAMLError:
        return None, text


# ── Search ────────────────────────────────────────────────────────

_SEARCH_CACHE: dict[tuple, list[dict]] = {}
_MAX_CACHE = 20
_SEARCH_CACHE_FILE = CACHE_DIR / ".okf-nav-search-cache.json"


def _load_search_cache() -> None:
    if _SEARCH_CACHE_FILE.exists():
        try:
            data = json.loads(_SEARCH_CACHE_FILE.read_text())
            for k_str, v in data.items():
                key = tuple(json.loads(k_str))
                _SEARCH_CACHE[key] = v
        except Exception:
            pass


def _save_search_cache() -> None:
    serializable = {json.dumps(k, ensure_ascii=False): v for k, v in _SEARCH_CACHE.items()}
    _SEARCH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SEARCH_CACHE_FILE.write_text(json.dumps(serializable, ensure_ascii=False))


def _clear_search_cache() -> None:
    _SEARCH_CACHE.clear()
    _SEARCH_CACHE_FILE.unlink(missing_ok=True)


def _title_key(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r'\s*\([^)]*\)\s*', '', t)
    t = re.sub(r'\d{4}-\d{2}-\d{2}', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def deduplicate(results: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in results:
        key = _title_key(r["title"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    deduped = []
    for key, group in groups.items():
        group.sort(key=lambda r: -r["score"])
        winner = dict(group[0])
        winner["merge_count"] = len(group)
        deduped.append(winner)
    deduped.sort(key=lambda r: (-r["score"], r["bundle"], r["path"]))
    return deduped


def search(query: str, bundle_filter: str | None = None,
           limit: int = 20, dedup: bool = True,
           skip_cache: bool = False) -> list[dict]:
    """Full-text search across all OKF bundles."""
    query_lower = query.lower()
    cache_key = (query_lower, bundle_filter, limit)
    if not skip_cache:
        _load_search_cache()
        if cache_key in _SEARCH_CACHE:
            return _SEARCH_CACHE[cache_key]

    terms = query_lower.split()
    results: list[dict] = []

    bundles = get_bundles()
    for bname, bpath in bundles.items():
        if bundle_filter and bname != bundle_filter:
            continue
        for md_path in sorted(bpath.rglob("*.md")):
            if md_path.name in ("index.md", "log.md"):
                continue
            rel = str(md_path.relative_to(bpath))
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            fm, body = parse_frontmatter(text)
            if not isinstance(fm, dict):
                continue

            ttype = fm.get("type", "")
            title = fm.get("title", "") or md_path.stem
            desc = fm.get("description", "") or ""
            tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tags_str = " ".join(str(t) for t in tags)

            search_text = f"{ttype} {title} {desc} {tags_str} {body[:2000]}".lower()
            fm_text = f"{ttype} {title} {desc} {tags_str}".lower()
            fm_matches = sum(1 for t in terms if t in fm_text)
            body_matches = sum(1 for t in terms if t in search_text)
            if fm_matches == 0 and body_matches == 0:
                continue

            results.append({
                "score": fm_matches * 3 + body_matches,
                "bundle": bname,
                "path": rel,
                "type": ttype,
                "title": title[:80],
                "description": desc[:120],
                "tags": list(tags) if isinstance(tags, list) else [str(tags)],
                "matched": fm_matches > 0,
            })

    results.sort(key=lambda r: (-r["score"], r["bundle"], r["path"]))
    if dedup:
        results = deduplicate(results)
    _SEARCH_CACHE[cache_key] = results[:limit]
    if len(_SEARCH_CACHE) > _MAX_CACHE:
        _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))
    _save_search_cache()
    return results[:limit]


# ── Semantic search (TF-IDF) ──────────────────────────────────────

TFIDF_CACHE = CACHE_DIR / ".okf-nav-tfidf"


def _collect_documents() -> list[dict]:
    docs: list[dict] = []
    for bname, bpath in get_bundles().items():
        for md_path in sorted(bpath.rglob("*.md")):
            if md_path.name in ("index.md", "log.md"):
                continue
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            fm, body = parse_frontmatter(text)
            if not isinstance(fm, dict):
                continue
            title = fm.get("title", md_path.stem) or md_path.stem
            desc = fm.get("description", "") or ""
            tags = fm.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tags_str = " ".join(str(t) for t in tags)
            docs.append({
                "bundle": bname,
                "path": str(md_path.relative_to(bpath)),
                "type": fm.get("type", ""),
                "title": title[:80],
                "description": desc[:120],
                "tags": tags if isinstance(tags, list) else [str(tags)],
                "searchable": f"{title} {desc} {tags_str} {body[:3000]}",
            })
    return docs


def _build_tfidf_index(docs: list[dict]) -> tuple:
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english",
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([d["searchable"] for d in docs])
    return vectorizer, matrix


def _save_tfidf_index(vectorizer, matrix, docs) -> None:
    TFIDF_CACHE.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_CACHE / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(TFIDF_CACHE / "matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)
    with open(TFIDF_CACHE / "docs.pkl", "wb") as f:
        pickle.dump(docs, f)


def _load_tfidf_index():
    try:
        if not (TFIDF_CACHE / "vectorizer.pkl").exists():
            return None
        with open(TFIDF_CACHE / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(TFIDF_CACHE / "matrix.pkl", "rb") as f:
            matrix = pickle.load(f)
        with open(TFIDF_CACHE / "docs.pkl", "rb") as f:
            docs = pickle.load(f)
        return vectorizer, matrix, docs
    except Exception:
        return None


def semantic_search(query: str, limit: int = 20) -> list[dict]:
    """TF-IDF semantic search across all bundles."""
    if not _HAS_TFIDF:
        print(f"ERROR: scikit-learn not available ({_TFIDF_ERR})", file=sys.stderr)
        print("  Install with: pip install 'okf-nav[semantic]' or uv tool install 'okf-nav[semantic]'", file=sys.stderr)
        return []

    cached = _load_tfidf_index()
    if cached:
        vectorizer, matrix, docs = cached
    else:
        print("  Building TF-IDF index (first run)...", file=sys.stderr)
        docs = _collect_documents()
        if not docs:
            return []
        vectorizer, matrix = _build_tfidf_index(docs)
        _save_tfidf_index(vectorizer, matrix, docs)

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    scored = [(i, score) for i, score in enumerate(scores) if score > 0]
    scored.sort(key=lambda x: -x[1])

    results = []
    for i, score in scored[:limit]:
        d = docs[i]
        results.append({
            "score": round(float(score), 4),
            "bundle": d["bundle"], "path": d["path"],
            "type": d["type"], "title": d["title"],
            "description": d["description"], "tags": d["tags"],
            "matched": True,
        })
    return results


def rebuild_tfidf_index() -> None:
    if not _HAS_TFIDF:
        return
    docs = _collect_documents()
    if docs:
        vectorizer, matrix = _build_tfidf_index(docs)
        _save_tfidf_index(vectorizer, matrix, docs)
        print(f"  TF-IDF index rebuilt: {len(docs)} docs, {matrix.shape[1]} features")


# ── Show ──────────────────────────────────────────────────────────

def show(path: str) -> dict | None:
    """Show a concept doc by bundle-relative path."""
    for bname, bpath in get_bundles().items():
        target = bpath / path
        if target.exists() and target.is_file():
            text = target.read_text(encoding="utf-8", errors="replace")
            fm, body = parse_frontmatter(text)
            return {
                "bundle": bname, "path": path,
                "frontmatter": fm if isinstance(fm, dict) else {},
                "body": body.strip(),
            }
    return None


# ── Status ────────────────────────────────────────────────────────

def status() -> None:
    """Show bundle overview."""
    bundles = get_bundles()
    if not bundles:
        print("No OKF bundles found.")
        return

    print(f"{'Bundle':25s} {'Concepts':>9s} {'Indices':>8s} {'Size':>8s}  Types")
    print("-" * 80)
    total_concepts = total_indices = total_size = 0
    all_types: Counter = Counter()

    for bname, bpath in sorted(bundles.items()):
        concepts = len([f for f in bpath.rglob("*.md") if f.name not in ("index.md", "log.md")])
        indices = len(list(bpath.rglob("index.md")))
        size = sum(f.stat().st_size for f in bpath.rglob("*.md"))
        size_str = f"{size/1024:.0f}K" if size < 1024 * 1024 else f"{size/1024/1024:.1f}M"

        type_counts: Counter = Counter()
        for md in bpath.rglob("*.md"):
            if md.name in ("index.md", "log.md"):
                continue
            text = md.read_text(encoding="utf-8", errors="replace")
            fm, _ = parse_frontmatter(text)
            if isinstance(fm, dict) and fm.get("type"):
                type_counts[fm["type"]] += 1
        type_str = ", ".join(f"{t}={c}" for t, c in type_counts.most_common(5))
        all_types.update(type_counts)

        print(f"{bname:25s} {concepts:>9d} {indices:>8d} {size_str:>8s}  {type_str}")
        total_concepts += concepts
        total_indices += indices
        total_size += size

    total_size_str = f"{total_size/1024:.0f}K" if total_size < 1024 * 1024 else f"{total_size/1024/1024:.1f}M"
    print("-" * 80)
    print(f"{'TOTAL':25s} {total_concepts:>9d} {total_indices:>8d} {total_size_str:>8s}")
    print(f"\nType distribution:")
    for t, c in all_types.most_common():
        print(f"  {t}: {c}")


# ── Topics ────────────────────────────────────────────────────────

def topics() -> None:
    """List all bundles with hierarchical structure."""
    for bname, bpath in sorted(get_bundles().items()):
        print(f"\n📁 {bname}/")
        for sub in sorted(bpath.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                concepts = len([f for f in sub.rglob("*.md") if f.name not in ("index.md", "log.md")])
                print(f"  ├── {sub.name}/ ({concepts} concepts)")
        root_mds = sorted(f for f in bpath.iterdir() if f.suffix == ".md" and f.name not in ("index.md", "log.md"))
        for mf in root_mds:
            text = mf.read_text(encoding="utf-8", errors="replace")
            title = mf.stem
            fm, _ = parse_frontmatter(text)
            if isinstance(fm, dict):
                title = fm.get("title", title)
            print(f"  ├── {title[:60]}")


# ── Rebuild indices ───────────────────────────────────────────────

def rebuild_indices(bundle_filter: str | None = None) -> None:
    """Rebuild index.md for bundles."""
    INDEX_MARKER = CACHE_DIR / ".okf-nav-index-timestamp"
    last_index = INDEX_MARKER.stat().st_mtime if INDEX_MARKER.exists() else 0
    _clear_search_cache()

    bundles = get_bundles()
    if bundle_filter:
        bundles = {k: v for k, v in bundles.items() if k == bundle_filter}

    def needs_rebuild(bname: str, bpath: Path) -> bool:
        if bundle_filter is not None:
            return True
        for md in bpath.rglob("*.md"):
            if md.name in ("index.md", "log.md"):
                continue
            if md.stat().st_mtime > last_index:
                return True
        return False

    def generate_index(bundle_root: Path) -> None:
        for dirpath, dirnames, filenames in os.walk(bundle_root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            rel_dir = os.path.relpath(dirpath, bundle_root)
            if rel_dir == ".":
                rel_dir = ""
            md_files = sorted(f for f in filenames if f.endswith(".md") and f not in ("index.md", "log.md"))
            subdirs = sorted(dirnames)
            if not md_files and not subdirs:
                continue
            idx_path = os.path.join(dirpath, "index.md")
            if os.path.exists(idx_path):
                existing = open(idx_path, encoding="utf-8").read()
                if "okf_version" in existing[:500]:
                    continue
            section = rel_dir.replace("/", " / ") if rel_dir else "Knowledge Base"
            lines = [f"# {section}\n"]
            for sd in subdirs:
                lines.append(f"* [{sd}/]({sd}/)\n")
            for mf in md_files:
                path = os.path.join(dirpath, mf)
                with open(path, encoding="utf-8") as fh:
                    c = fh.read()
                fm, _ = parse_frontmatter(c)
                if isinstance(fm, dict):
                    title = fm.get("title", mf.replace(".md", ""))
                    desc = fm.get("description", "")
                    label = f"{title} — {desc}" if desc else title
                    lines.append(f"* [{label}]({mf})\n")
                else:
                    lines.append(f"* [{mf.replace('.md', '')}]({mf})\n")
            with open(idx_path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            print(f"  ✓ {rel_dir or '.'}")

    for bname, bpath in bundles.items():
        if not needs_rebuild(bname, bpath):
            print(f"  [skip] {bname}: unchanged")
            continue
        print(f"\nRebuilding {bname}...")
        generate_index(bpath)

    INDEX_MARKER.parent.mkdir(parents=True, exist_ok=True)
    INDEX_MARKER.write_text(str(time.time()))


# ── Context hook ──────────────────────────────────────────────────

def context(prompt: str, max_results: int = 5) -> str:
    """Auto-context: extract keywords, search, return pointer block."""
    if len(prompt) < 15 or len(prompt.split()) < 3:
        return ""

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "that", "this", "these", "those",
        "it", "its", "my", "your", "our", "their", "his", "her",
        "什么", "怎么", "如何", "为什么", "哪个", "这个", "那个",
        "我们", "你们", "他们", "它们", "的", "了", "在", "是", "我",
        "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
        "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看",
        "好", "自己", "这",
    }
    words = re.findall(r'[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}', prompt.lower())
    keywords = [w for w in words if w not in stopwords]
    if not keywords:
        return ""

    results = search(" ".join(keywords[:8]), limit=max_results)
    if not results:
        return ""

    def _snippet(bundle: str, path: str, max_c: int = 400) -> str:
        bp = get_bundles().get(bundle, BUNDLES_DIR / bundle) / path
        if not bp.exists():
            return ""
        try:
            text = bp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        _, body = parse_frontmatter(text)
        body = re.sub(r'^#+\s+', '', body, flags=re.MULTILINE)
        body = re.sub(r'\n{3,}', '\n\n', body).strip()
        if len(body) <= max_c:
            return body
        truncated = body[:max_c]
        last_space = truncated.rfind(" ")
        if last_space > max_c // 2:
            truncated = truncated[:last_space]
        return truncated + "…"

    top = results[0]
    snippet = _snippet(top["bundle"], top["path"])
    lines = [
        "<!-- okf-nav:context -->",
        "Relevant knowledge from personal knowledge base:",
    ]
    if snippet:
        lines.append("")
        lines.append(f"📖 [{top['title']}]({top['bundle']}/{top['path']})")
        lines.append("```")
        lines.append(snippet)
        lines.append("```")
        lines.append(f"   (全文: `okf-nav show {top['bundle']}/{top['path']}`)")
        remaining = results[1:max_results]
        if remaining:
            lines.append("")
            lines.append("Related:")
            for r in remaining:
                tag_str = f" [{', '.join(r['tags'][:3])}]" if r.get("tags") else ""
                lines.append(f"  • [{r['title']}]({r['bundle']}/{r['path']}) — [{r['type']}]{tag_str}")
    else:
        for r in results:
            tag_str = f" [{', '.join(r['tags'][:3])}]" if r.get("tags") else ""
            lines.append(f"  • [{r['title']}]({r['bundle']}/{r['path']}) — [{r['type']}]{tag_str}")
    lines.append(f"  (search: `okf-nav search {' '.join(keywords[:6])}`)")
    lines.append("<!-- /okf-nav:context -->")
    return "\n".join(lines)


# ── Health ────────────────────────────────────────────────────────

def health() -> list:
    """Run comprehensive health check."""
    issues: list[tuple[str, str]] = []
    ok_count = 0

    # Bundle quality
    bundles_dir = BUNDLES_DIR.resolve() if not BUNDLES_DIR.is_absolute() else BUNDLES_DIR
    missing_type = missing_desc = untagged = total = 0
    types: dict[str, int] = {}
    for p in sorted(bundles_dir.rglob("*.md")):
        if p.name in ("index.md", "log.md"):
            continue
        if ".archive" in str(p):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fm, _ = parse_frontmatter(text)
        if not isinstance(fm, dict):
            continue
        if "type" not in fm:
            missing_type += 1
        else:
            t = fm["type"]
            types[t] = types.get(t, 0) + 1
        if not fm.get("description"):
            missing_desc += 1
        if not fm.get("tags"):
            untagged += 1
        total += 1

    ok_count += 1
    if missing_type:
        issues.append(("ERROR", f"{missing_type} concepts missing 'type' field"))
    else:
        ok_count += 1
    if missing_desc:
        issues.append(("WARN", f"{missing_desc}/{total} concepts missing description"))
    else:
        ok_count += 1
    if untagged:
        issues.append(("WARN", f"{untagged}/{total} concepts untagged"))

    # Stale check
    stale_count = 0
    now_dt = datetime.now(timezone.utc)
    for p in sorted(bundles_dir.rglob("*.md")):
        if p.name in ("index.md", "log.md"):
            continue
        if ".archive" in str(p):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fm, _ = parse_frontmatter(text)
        if not isinstance(fm, dict):
            continue
        ts = fm.get("timestamp", "")
        if not ts:
            continue
        try:
            ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            if (now_dt - ts_dt).days >= 90:
                stale_count += 1
        except Exception:
            continue
    if stale_count:
        issues.append(("WARN", f"{stale_count} stale entries (>=90d, run 'okf-nav stale --archive')"))

    # Type diversity
    if len(types) > 15:
        issues.append(("INFO", f"{len(types)} distinct types (consider consolidating)"))

    # Report
    print("=== okf-nav health ===\n")
    errors = [i for i in issues if i[0] == "ERROR"]
    warns = [i for i in issues if i[0] == "WARN"]
    infos = [i for i in issues if i[0] == "INFO"]
    if errors:
        print(f"Errors ({len(errors)}):")
        for _, msg in errors:
            print(f"  ❌ {msg}")
        print()
    if warns:
        print(f"Warnings ({len(warns)}):")
        for _, msg in warns:
            print(f"  ⚠️  {msg}")
        print()
    if infos:
        print(f"Info ({len(infos)}):")
        for _, msg in infos:
            print(f"  ℹ️  {msg}")
        print()
    if not errors:
        print(f"✅ All clean ({ok_count} checks passed)\n")
    print(f"Summary: {total} concepts, {len(types)} types, {len(issues)} issue(s)")
    return issues


# ── Audit + Fix ───────────────────────────────────────────────────

def audit(fix: bool = False) -> list:
    """Run health check and optionally auto-fix issues."""
    issues = health()
    if not fix:
        return issues

    bundles_dir = BUNDLES_DIR.resolve() if not BUNDLES_DIR.is_absolute() else BUNDLES_DIR
    fix_count = archive_count = 0

    print("\n=== Auto-fix ===\n")

    for p in sorted(bundles_dir.rglob("*.md")):
        if p.name in ("index.md", "log.md"):
            continue
        if ".archive" in str(p):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(text)
        if not isinstance(fm, dict):
            continue
        rel = str(p.relative_to(bundles_dir))
        changed = False

        # Fix missing description
        if not fm.get("description") and body:
            desc = re.sub(r"^#+\s+", "", body, flags=re.MULTILINE)
            desc = re.sub(r"\n+", " ", desc).strip()[:200]
            desc = desc.rstrip(",;")
            if desc:
                print(f"  +description → {rel}")
                fm["description"] = desc + ("…" if len(body) > 200 else "")
                changed = True

        # Fix missing tags
        if not fm.get("tags"):
            inferred = []
            ttype = fm.get("type", "")
            if ttype:
                inferred.append(ttype.lower().replace(" ", "-"))
            path_parts = Path(rel).parts
            if len(path_parts) > 1 and path_parts[0] not in inferred:
                inferred.append(path_parts[0])
            if inferred:
                print(f"  +tags: {inferred} → {rel}")
                fm["tags"] = inferred
                changed = True

        if changed:
            new_fm = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
            p.write_text(f"---\n{new_fm}\n---\n\n{body}\n", encoding="utf-8")
            fix_count += 1

    # Auto-archive stale
    now_dt = datetime.now(timezone.utc)
    for p in sorted(bundles_dir.rglob("*.md")):
        if p.name in ("index.md", "log.md"):
            continue
        if ".archive" in str(p):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        fm, _ = parse_frontmatter(text)
        if not isinstance(fm, dict):
            continue
        ts = fm.get("timestamp", "")
        if not ts:
            continue
        try:
            ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            if (now_dt - ts_dt).days >= 90:
                rel = str(p.relative_to(bundles_dir))
                archive_dir = p.parent / ".archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(archive_dir / p.name))
                print(f"  archived (stale) → {rel}")
                archive_count += 1
        except Exception:
            continue

    print(f"\n✓ Fixed: {fix_count} concept(s), archived {archive_count} stale")
    if fix_count or archive_count:
        rebuild_indices()
    return issues


# ── Export JSON ───────────────────────────────────────────────────

def export_json(bundle_filter: str | None = None, minify: bool = False) -> None:
    """Export bundles as JSON."""
    bundles = get_bundles()
    if bundle_filter:
        if bundle_filter not in bundles:
            print(f"ERROR: bundle '{bundle_filter}' not found", file=sys.stderr)
            print(f"  Available: {', '.join(bundles.keys())}", file=sys.stderr)
            sys.exit(1)
        bundles = {bundle_filter: bundles[bundle_filter]}

    result: dict[str, list[dict]] = {}
    for bname, bpath in sorted(bundles.items()):
        concepts: list[dict] = []
        for md_path in sorted(bpath.rglob("*.md")):
            if md_path.name in ("index.md", "log.md"):
                continue
            if ".archive" in str(md_path):
                continue
            text = md_path.read_text(encoding="utf-8", errors="replace")
            rel = str(md_path.relative_to(bpath))
            fm, body = parse_frontmatter(text)
            concept: dict = {"path": rel, "body": body.strip()}
            if isinstance(fm, dict):
                concept["frontmatter"] = fm
            concepts.append(concept)
        result[bname] = concepts

    print(json.dumps(result, ensure_ascii=False, indent=None if minify else 2))


# ── Update ────────────────────────────────────────────────────────

def update_concept(path_str: str, sets: list[str], add_tags: list[str],
                   remove_tags: list[str]) -> int:
    """Edit frontmatter fields of a concept."""
    bundles_dir = BUNDLES_DIR.resolve() if not BUNDLES_DIR.is_absolute() else BUNDLES_DIR
    target = Path(path_str)
    if not target.is_absolute():
        target = bundles_dir / target
    if not target.exists():
        print(f"ERROR: not found: {target}")
        return 1

    text = target.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_frontmatter(text)
    if not isinstance(fm, dict):
        print("ERROR: not an OKF concept (no frontmatter)")
        return 1

    modified = False
    for s in sets:
        if "=" not in s:
            print(f"WARN: ignoring '{s}' (format: key=value)")
            continue
        key, val = s.split("=", 1)
        key, val = key.strip(), val.strip()
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.isdigit():
            val = int(val)
        elif val.replace(".", "", 1).isdigit():
            val = float(val)
        old = fm.get(key)
        fm[key] = val
        print(f"  {key}: {old!r} → {val!r}")
        modified = True

    for t in add_tags:
        existing = fm.get("tags", [])
        if isinstance(existing, str):
            existing = [existing]
        if t in existing:
            print(f"  tag already exists: {t}")
            continue
        existing.append(t)
        fm["tags"] = existing
        print(f"  +tag: {t}")
        modified = True

    for t in remove_tags:
        existing = fm.get("tags", [])
        if isinstance(existing, str):
            existing = [existing]
        if t not in existing:
            print(f"  tag not found: {t}")
            continue
        existing.remove(t)
        fm["tags"] = existing
        print(f"  -tag: {t}")
        modified = True

    if not modified:
        print("No changes made.")
        return 0

    new_fm = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    target.write_text(f"---\n{new_fm}\n---\n\n{body}\n", encoding="utf-8")
    rel = str(target.relative_to(bundles_dir)) if target.is_relative_to(bundles_dir) else str(target)
    print(f"✓ Updated: {rel}")
    return 0


# ── Stale ─────────────────────────────────────────────────────────

def stale(older_days: int = 90, bundle_filter: str | None = None,
          type_filter: str | None = None, archive: bool = False) -> list[dict]:
    """List and optionally archive stale entries."""
    bundles_dir = BUNDLES_DIR.resolve() if not BUNDLES_DIR.is_absolute() else BUNDLES_DIR
    results: list[dict] = []
    now = datetime.now(timezone.utc)

    bundles = get_bundles()
    for bname, bpath in bundles.items():
        if bundle_filter and bname != bundle_filter:
            continue
        for md_path in sorted(bpath.rglob("*.md")):
            if md_path.name in ("index.md", "log.md"):
                continue
            if ".archive" in str(md_path):
                continue
            text = md_path.read_text(encoding="utf-8", errors="replace")
            fm, _ = parse_frontmatter(text)
            if not isinstance(fm, dict):
                continue
            ttype = fm.get("type", "")
            if type_filter and ttype != type_filter:
                continue
            ts_str = fm.get("timestamp", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            age_days = (now - ts).days
            if age_days < older_days:
                continue
            title = fm.get("title", md_path.stem)[:60]
            results.append({
                "bundle": bname, "path": str(md_path.relative_to(bpath)),
                "type": ttype, "title": title,
                "timestamp": str(ts_str)[:10], "age_days": age_days,
            })

    results.sort(key=lambda r: -r["age_days"])
    if results:
        print(f"Stale entries (≥{older_days} days without update):\n")
        for r in results:
            print(f"  [{r['type']:20s}] {r['title']}")
            print(f"          {r['bundle']}/{r['path']}")
            print(f"          Last: {r['timestamp']} ({r['age_days']}d ago)\n")
        print(f"Total: {len(results)} stale entries")

        if archive:
            yn = input(f"\nArchive {len(results)} stale file(s) to .archive/? [y/N] ").strip().lower()
            if yn == "y":
                archived = 0
                for r in results:
                    bpath = bundles.get(r["bundle"])
                    if not bpath:
                        continue
                    full = bpath / r["path"]
                    if not full.exists():
                        continue
                    archive_dir = full.parent / ".archive"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(full), str(archive_dir / full.name))
                    print(f"  ARCHIVED {r['bundle']}/{r['path']}")
                    archived += 1
                print(f"\nArchived: {archived} file(s) moved to .archive/.")
                rebuild_indices()
    else:
        print(f"No stale entries found (threshold: {older_days}d).")
    return results


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return

    cmd = sys.argv[1]

    if cmd == "search":
        use_all = "--all" in sys.argv
        semantic = "--semantic" in sys.argv or "-s" in sys.argv
        query = " ".join(a for a in sys.argv[2:] if not a.startswith("--"))
        if not query:
            print("Usage: okf-nav search <query> [--all] [--semantic|-s]")
            return
        if semantic:
            results = semantic_search(query, limit=40 if use_all else 20)
        else:
            results = search(query, dedup=not use_all)
        if not results:
            print("No results found.")
            return
        print(f"Found {len(results)} result(s):\n")
        for r in results:
            fm_tag = " ✅" if r.get("matched") else ""
            tag_str = f" [{', '.join(r['tags'][:3])}]" if r.get("tags") else ""
            merge_info = f" (+{r['merge_count']-1} similar)" if r.get("merge_count", 1) > 1 else ""
            print(f"  [{r['type']:20s}] {r['title']}{tag_str}{merge_info}")
            print(f"          {r['bundle']}/{r['path']}")
            if r["description"]:
                print(f"          {r['description'][:100]}")
            print()
        if not use_all:
            print("  (use --all to show all results including duplicates)")

    elif cmd == "show":
        path = " ".join(sys.argv[2:])
        if not path:
            print("Usage: okf-nav show <path> (e.g. 'my-bundle/concepts/foo.md')")
            return
        doc = show(path)
        if not doc:
            results = search(path, limit=5)
            if results:
                print(f"Path not found. Did you mean:\n")
                for r in results:
                    print(f"  {r['bundle']}/{r['path']}  [{r['type']}] {r['title']}")
            else:
                print(f"Not found: {path}")
            return
        print(f"── {doc['bundle']}/{doc['path']} ──")
        if doc["frontmatter"]:
            print(f"Type: {doc['frontmatter'].get('type', '?')}")
            print(f"Title: {doc['frontmatter'].get('title', '?')}")
            if doc["frontmatter"].get("description"):
                print(f"Description: {doc['frontmatter']['description']}")
            tags = doc["frontmatter"].get("tags", [])
            if tags:
                tag_str = ", ".join(tags[:5]) if isinstance(tags, list) else str(tags)
                print(f"Tags: {tag_str}")
            print()
        print(doc.get("body", ""))

    elif cmd == "index":
        rebuild_indices()

    elif cmd == "status":
        status()

    elif cmd == "topics":
        topics()

    elif cmd == "context":
        prompt = " ".join(sys.argv[2:])
        if not prompt:
            print("Usage: okf-nav context <prompt>")
            return
        block = context(prompt)
        if block:
            print(block)

    elif cmd == "health":
        health()

    elif cmd == "audit":
        fix = "--fix" in sys.argv
        audit(fix=fix)

    elif cmd == "export":
        if "--json" not in sys.argv:
            print("Usage: okf-nav export --json [--bundle x] [--minify]")
            return
        bundle_filter = None
        minify = "--minify" in sys.argv
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--bundle" and i + 1 < len(args):
                bundle_filter = args[i + 1]
                i += 2
            else:
                i += 1
        export_json(bundle_filter=bundle_filter, minify=minify)

    elif cmd == "update":
        args = sys.argv[2:]
        if not args:
            print("Usage: okf-nav update <path> [--set k=v] [--add-tag t] [--remove-tag t]")
            return
        path_str = args[0]
        sets: list[str] = []
        add_tags: list[str] = []
        remove_tags: list[str] = []
        i = 1
        while i < len(args):
            if args[i] == "--set" and i + 1 < len(args):
                sets.append(args[i + 1])
                i += 2
            elif args[i] == "--add-tag" and i + 1 < len(args):
                add_tags.append(args[i + 1])
                i += 2
            elif args[i] == "--remove-tag" and i + 1 < len(args):
                remove_tags.append(args[i + 1])
                i += 2
            else:
                i += 1
        update_concept(path_str, sets, add_tags, remove_tags)

    elif cmd == "stale":
        older = 90
        bundle = None
        ttype = None
        archive_flag = False
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--older" and i + 1 < len(args):
                older = int(args[i + 1])
                i += 2
            elif args[i] == "--bundle" and i + 1 < len(args):
                bundle = args[i + 1]
                i += 2
            elif args[i] == "--type" and i + 1 < len(args):
                ttype = args[i + 1]
                i += 2
            elif args[i] == "--archive":
                archive_flag = True
                i += 1
            else:
                i += 1
        stale(older_days=older, bundle_filter=bundle,
              type_filter=ttype, archive=archive_flag)

    elif cmd in ("-h", "--help"):
        print(__doc__.strip())

    elif cmd in ("-V", "--version"):
        from okf_nav import __version__
        print(f"okf-nav {__version__}")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__.strip())


if __name__ == "__main__":
    main()
