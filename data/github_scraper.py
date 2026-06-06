"""
GitHub PR review comment scraper with LLM-based semi-automatic classification.

Fetches real code review comments from GitHub, classifies them using Claude Haiku
(or keyword fallback), and merges with existing dataset splits.

Usage:
    venv311/bin/python data/github_scraper.py --dry-run
    venv311/bin/python data/github_scraper.py --repos "django/django" --max-prs 20
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

SPLITS_DIR = Path(__file__).parent / "splits"

VALID_LABELS = ["security", "architecture", "observability", "style", "false_positive"]

LABEL_KEYWORDS = {
    "security": [
        "sql injection", "xss", "csrf", "authentication", "authorization",
        "password", "token", "secret", "credential", "sanitize", "escape",
        "vulnerability", "exploit", "privilege", "injection", "unsafe",
        "hardcoded", "plaintext", "encrypt", "hash", "tls", "ssl", "cert",
        "input validation", "deserialization", "path traversal", "rce",
    ],
    "architecture": [
        "coupling", "cohesion", "dependency", "interface", "abstraction",
        "single responsibility", "open closed", "solid", "design pattern",
        "separation of concerns", "modularity", "encapsulation", "inheritance",
        "composition", "refactor", "layer", "domain", "bounded context",
        "god class", "circular", "tight coupling", "responsibility",
    ],
    "observability": [
        "logging", "log", "tracing", "trace", "metric", "monitor",
        "instrumentation", "observable", "debug", "audit", "event",
        "correlation id", "structured log", "span", "telemetry", "alert",
        "dashboard", "health check", "error tracking",
    ],
    "style": [
        "naming", "variable name", "function name", "method name",
        "readability", "formatting", "indent", "whitespace", "comment",
        "documentation", "docstring", "typo", "consistency", "convention",
        "style", "clean code", "magic number", "magic string", "dead code",
        "unused", "redundant",
    ],
}


# ---------------------------------------------------------------------------
# Keyword-based fallback classifier (mirrors prepare_dataset.py logic)
# ---------------------------------------------------------------------------

def _classify_keyword(text: str) -> str:
    """Keyword-based classification. Falls back to 'style' if ambiguous."""
    text_lower = text.lower()
    scores: dict[str, int] = defaultdict(int)

    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[label] += 1

    if not scores:
        return "style"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_score = ranked[0][1]
    top_labels = [lbl for lbl, s in ranked if s == best_score]

    return top_labels[0]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request_with_retry(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> requests.Response:
    """GET with exponential backoff on transient errors."""
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            return resp
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise
            wait = delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] {exc} — aguardando {wait:.0f}s")
            time.sleep(wait)
    # unreachable, mypy
    raise RuntimeError("unreachable")


def _handle_rate_limit(resp: requests.Response) -> None:
    """If GitHub rate limit is nearly exhausted, sleep until reset."""
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
    if remaining < 5:
        reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
        now = time.time()
        wait = max(0, reset_ts - now) + 2
        print(f"  Rate limit quase esgotado (remaining={remaining}) — aguardando {wait:.0f}s")
        time.sleep(wait)


def _is_bot(login: str) -> bool:
    low = login.lower()
    return "bot" in low or "[bot]" in low


# ---------------------------------------------------------------------------
# 1. fetch_pr_comments
# ---------------------------------------------------------------------------

def fetch_pr_comments(
    repo: str,
    max_prs: int = 50,
    token: str | None = None,
) -> list[dict]:
    """
    Fetch PR review comments from a GitHub repository.

    Returns list of {"text", "repo", "pr", "url"}.
    """
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")

    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base_url = "https://api.github.com"
    collected_comments: list[dict] = []
    prs_fetched = 0
    page = 1

    print(f"\n[{repo}] Buscando PRs fechados...")

    while prs_fetched < max_prs:
        per_page = min(20, max_prs - prs_fetched)
        try:
            resp = _request_with_retry(
                f"{base_url}/repos/{repo}/pulls",
                headers=headers,
                params={"state": "closed", "per_page": per_page, "page": page},
            )
        except requests.exceptions.RequestException as exc:
            print(f"  Erro ao buscar PRs página {page}: {exc}")
            break

        _handle_rate_limit(resp)

        if resp.status_code == 404:
            print(f"  Repositório não encontrado: {repo}")
            break
        if resp.status_code == 403:
            print(f"  Acesso negado (403). Token pode ser necessário.")
            break
        if resp.status_code != 200:
            print(f"  Erro HTTP {resp.status_code} ao buscar PRs")
            break

        prs = resp.json()
        if not prs:
            break

        for pr in prs:
            pr_number = pr["number"]
            prs_fetched += 1

            try:
                comments_resp = _request_with_retry(
                    f"{base_url}/repos/{repo}/pulls/{pr_number}/comments",
                    headers=headers,
                    params={"per_page": 100},
                )
            except requests.exceptions.RequestException as exc:
                print(f"    PR #{pr_number} — erro ao buscar comments: {exc}")
                continue

            _handle_rate_limit(comments_resp)

            if comments_resp.status_code != 200:
                continue

            for comment in comments_resp.json():
                body: str = comment.get("body", "") or ""
                login: str = comment.get("user", {}).get("login", "") or ""

                if len(body) < 30 or len(body) > 600:
                    continue
                if _is_bot(login):
                    continue

                collected_comments.append({
                    "text": body.strip(),
                    "repo": repo,
                    "pr": pr_number,
                    "url": comment.get("html_url", ""),
                })

        page += 1

    print(f"  [{repo}] {prs_fetched} PRs → {len(collected_comments)} comments coletados")
    return collected_comments


# ---------------------------------------------------------------------------
# 2. classify_with_llm
# ---------------------------------------------------------------------------

def _build_classification_prompt(batch: list[dict]) -> str:
    items_text = "\n".join(
        f'{i + 1}. """{item["text"]}"""'
        for i, item in enumerate(batch)
    )
    labels_str = ", ".join(VALID_LABELS)
    return f"""You are a code review classifier. Classify each code review comment into exactly one of these labels: {labels_str}.

Definitions:
- security: vulnerabilities, auth issues, injection, data exposure, cryptography
- architecture: coupling, cohesion, SOLID, design patterns, layer violations, DDD
- observability: missing logging, tracing, metrics, monitoring, alerting
- style: naming, formatting, readability, comments, dead code, conventions
- false_positive: the finding is incorrect, intentional, or not applicable

For each numbered comment below, respond with a JSON array where each element is:
{{"index": <number>, "label": "<label>", "confidence": "<high|medium|low>"}}

Comments:
{items_text}

Respond ONLY with valid JSON array, no explanation."""


def classify_with_llm(
    comments: list[dict],
    api_key: str | None = None,
) -> list[dict]:
    """
    Classify comments using Claude Haiku. Falls back to keyword classification
    if API key is unavailable.

    Returns comments with "label" field added, excluding low-confidence ones.
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    use_llm = bool(api_key)

    if not use_llm:
        print("  ANTHROPIC_API_KEY não encontrada — usando classificação por keywords")
        result = []
        for c in comments:
            classified = dict(c)
            classified["label"] = _classify_keyword(c["text"])
            classified["confidence"] = "medium"
            result.append(classified)
        return result

    try:
        import anthropic
    except ImportError:
        print("  Pacote 'anthropic' não instalado — usando classificação por keywords")
        result = []
        for c in comments:
            classified = dict(c)
            classified["label"] = _classify_keyword(c["text"])
            classified["confidence"] = "medium"
            result.append(classified)
        return result

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = anthropic.Anthropic(**client_kwargs)
    model = "claude-haiku-4-5-20251001"

    batch_size = 10
    classified_comments: list[dict] = []
    skipped_low = 0

    print(f"  Classificando {len(comments)} comments via LLM (batches de {batch_size})...")

    for batch_start in range(0, len(comments), batch_size):
        batch = comments[batch_start: batch_start + batch_size]
        prompt = _build_classification_prompt(batch)

        try:
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_content = message.content[0].text.strip()
        except Exception as exc:
            print(f"  Erro na API LLM (batch {batch_start}): {exc} — usando keyword fallback para este batch")
            for item in batch:
                classified = dict(item)
                classified["label"] = _classify_keyword(item["text"])
                classified["confidence"] = "medium"
                classified_comments.append(classified)
            continue

        # Parse JSON array from response
        classifications: list[dict] = []
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?", "", raw_content).strip().rstrip("```").strip()
            classifications = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt to extract JSON array with regex
            match = re.search(r"\[.*\]", raw_content, re.DOTALL)
            if match:
                try:
                    classifications = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not classifications:
            print(f"  Falha ao parsear JSON do LLM — keyword fallback para batch {batch_start}")
            for item in batch:
                classified = dict(item)
                classified["label"] = _classify_keyword(item["text"])
                classified["confidence"] = "medium"
                classified_comments.append(classified)
            continue

        # Map results back by index
        result_by_index: dict[int, dict] = {}
        for entry in classifications:
            if isinstance(entry, dict) and "index" in entry:
                result_by_index[int(entry["index"])] = entry

        for i, item in enumerate(batch):
            idx = i + 1
            entry = result_by_index.get(idx)

            if entry is None:
                # LLM missed this item — keyword fallback
                classified = dict(item)
                classified["label"] = _classify_keyword(item["text"])
                classified["confidence"] = "medium"
                classified_comments.append(classified)
                continue

            confidence = entry.get("confidence", "low").lower()
            label = entry.get("label", "").lower().strip()

            if label not in VALID_LABELS:
                # Fallback if label is invalid
                label = _classify_keyword(item["text"])
                confidence = "medium"

            if confidence == "low":
                skipped_low += 1
                continue

            classified = dict(item)
            classified["label"] = label
            classified["confidence"] = confidence
            classified_comments.append(classified)

        # Small delay to avoid hammering the API
        time.sleep(0.3)

    print(f"  Classificados: {len(classified_comments)}, descartados (low confidence): {skipped_low}")
    return classified_comments


# ---------------------------------------------------------------------------
# 3. merge_with_existing
# ---------------------------------------------------------------------------

def _load_splits() -> dict[str, list[dict]]:
    """Load existing JSONL splits from data/splits/."""
    splits: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        path = SPLITS_DIR / f"{split_name}.jsonl"
        examples: list[dict] = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        splits[split_name] = examples
    return splits


def _save_splits(splits: dict[str, list[dict]]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, examples in splits.items():
        path = SPLITS_DIR / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Salvo: {path} ({len(examples)} exemplos)")


def _stratified_split(
    examples: list[dict],
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified 80/10/10 split."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_label[ex["label"]].append(ex)

    train_all, val_all, test_all = [], [], []

    for label in sorted(by_label.keys()):
        items = by_label[label][:]
        random.shuffle(items)
        n = len(items)
        n_val = max(1, math.floor(n * val_ratio))
        n_test = max(1, math.floor(n * test_ratio))
        n_train = n - n_val - n_test

        train_all.extend(items[:n_train])
        val_all.extend(items[n_train: n_train + n_val])
        test_all.extend(items[n_train + n_val:])

    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    return train_all, val_all, test_all


def merge_with_existing(
    new_examples: list[dict],
    max_per_class: int = 200,
) -> dict:
    """
    Merge new examples with existing splits, rebalance, re-split, and save.

    Returns stats dict with "added", "total", "by_class".
    """
    existing_splits = _load_splits()

    # Combine all existing examples into one pool
    existing_pool: list[dict] = []
    for split_examples in existing_splits.values():
        existing_pool.extend(split_examples)

    existing_texts = {ex["text"] for ex in existing_pool}

    # Deduplicate new examples against existing pool
    fresh = []
    for ex in new_examples:
        if ex["text"] not in existing_texts and "label" in ex:
            fresh.append({"text": ex["text"], "label": ex["label"]})
            existing_texts.add(ex["text"])

    added_count = len(fresh)
    all_examples = existing_pool + fresh

    # Cap per-class at max_per_class
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        by_label[ex["label"]].append(ex)

    balanced: list[dict] = []
    by_class_final: dict[str, int] = {}
    for label in VALID_LABELS:
        items = by_label.get(label, [])
        random.shuffle(items)
        items = items[:max_per_class]
        balanced.extend(items)
        by_class_final[label] = len(items)

    train_split, val_split, test_split = _stratified_split(balanced)

    new_splits = {"train": train_split, "val": val_split, "test": test_split}
    _save_splits(new_splits)

    total = sum(by_class_final.values())
    return {
        "added": added_count,
        "total": total,
        "by_class": by_class_final,
    }


# ---------------------------------------------------------------------------
# 4. main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape GitHub PR review comments and enrich the classifier dataset."
    )
    parser.add_argument(
        "--repos",
        default="microsoft/vscode torvalds/linux django/django",
        help="Space-separated list of GitHub repos (owner/name)",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        default=30,
        help="Max PRs to fetch per repository",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM classification; use keyword fallback only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print examples but do not write to disk",
    )
    args = parser.parse_args()

    repos = args.repos.split()
    random.seed(42)

    print(f"Repositórios: {repos}")
    print(f"Max PRs por repo: {args.max_prs}")
    print(f"LLM: {'desabilitado (--no-llm)' if args.no_llm else 'habilitado'}")
    print(f"Dry-run: {args.dry_run}")

    # Step 1: fetch comments
    all_comments: list[dict] = []
    for repo in repos:
        try:
            comments = fetch_pr_comments(repo, max_prs=args.max_prs)
            all_comments.extend(comments)
        except Exception as exc:
            print(f"  Erro ao processar {repo}: {exc}")

    print(f"\nTotal de comments coletados: {len(all_comments)}")

    if not all_comments:
        print("Nenhum comment coletado. Encerrando.")
        sys.exit(0)

    # Step 2: classify
    if args.no_llm:
        classified = []
        for c in all_comments:
            entry = dict(c)
            entry["label"] = _classify_keyword(c["text"])
            entry["confidence"] = "medium"
            classified.append(entry)
        print(f"Classificação por keywords: {len(classified)} exemplos")
    else:
        classified = classify_with_llm(all_comments)

    if not classified:
        print("Nenhum exemplo após classificação. Encerrando.")
        sys.exit(0)

    # Print sample
    print("\n--- Amostra dos exemplos classificados (primeiros 5) ---")
    for ex in classified[:5]:
        label = ex.get("label", "?")
        confidence = ex.get("confidence", "?")
        text_preview = ex["text"][:120].replace("\n", " ")
        print(f"  [{label}] ({confidence}) {text_preview}")
    print("---")

    # Label distribution
    label_counts = Counter(ex["label"] for ex in classified)
    print("\nDistribuição por label:")
    for lbl in VALID_LABELS:
        print(f"  {lbl}: {label_counts.get(lbl, 0)}")

    if args.dry_run:
        print("\n[dry-run] Não salvando. Encerrando.")
        return

    # Step 3: merge and save
    print("\nMergeando com splits existentes...")
    stats = merge_with_existing(classified, max_per_class=200)

    print("\n=== Stats finais ===")
    print(f"  Novos exemplos adicionados: {stats['added']}")
    print(f"  Total no pool balanceado:   {stats['total']}")
    print("  Por classe:")
    for lbl, count in stats["by_class"].items():
        print(f"    {lbl}: {count}")


if __name__ == "__main__":
    main()
