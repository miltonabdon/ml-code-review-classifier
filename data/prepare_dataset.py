"""
Dataset preparation for ML Code Review Classifier.

Downloads alenphilip/Code-Review-Assistant from HuggingFace, maps fields
to 5 classification labels, balances classes, and creates stratified splits.

Labels:
  security      - vulnerabilities, auth issues, data exposure
  architecture  - coupling, cohesion, structural patterns
  observability - missing logging, tracing, metrics
  style         - naming, formatting, minor improvements
  false_positive - incorrect or non-applicable finding
"""

import json
import re
import os
import random
from collections import defaultdict, Counter
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "splits"

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


def classify_text(text: str) -> str | None:
    """Keyword-based label assignment. Returns None if ambiguous."""
    text_lower = text.lower()
    scores = defaultdict(int)

    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[label] += 1

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_score = ranked[0][1]
    top_labels = [l for l, s in ranked if s == best_score]

    if len(top_labels) == 1:
        return top_labels[0]
    return None


def build_finding_text(sample: dict) -> str | None:
    """Extract relevant text from a dataset sample for classification."""
    parts = []

    for field in ["comment", "review", "feedback", "suggestion", "description", "text", "output"]:
        val = sample.get(field, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    if not parts:
        for val in sample.values():
            if isinstance(val, str) and len(val) > 20:
                parts.append(val.strip())
                break

    return " ".join(parts)[:512] if parts else None


def load_hf_dataset() -> list[dict]:
    """Load dataset from HuggingFace. Falls back to synthetic generation if unavailable."""
    try:
        from datasets import load_dataset
        print("Baixando alenphilip/Code-Review-Assistant do HuggingFace...")
        ds = load_dataset("alenphilip/Code-Review-Assistant", split="train", trust_remote_code=True)
        print(f"  {len(ds)} exemplos carregados")
        print(f"  Campos: {list(ds.features.keys())}")
        return [dict(row) for row in ds]
    except Exception as e:
        print(f"HuggingFace indisponível ({e}). Usando dataset sintético local.")
        return generate_synthetic_dataset()


def generate_synthetic_dataset() -> list[dict]:
    """
    Generates a synthetic code review dataset with realistic findings.
    Used as fallback when HuggingFace is unavailable or for offline development.
    ~200 examples per class (1000 total) — enough to demonstrate train/val dynamics.
    """
    examples = {
        "security": [
            "SQL query built with string concatenation — use parameterized queries to prevent injection",
            "Password stored in plaintext — hash with bcrypt or argon2",
            "API key hardcoded in source — move to environment variable or secret manager",
            "No input validation before passing user data to shell command — command injection risk",
            "JWT secret is weak and predictable — use cryptographically random 256-bit key",
            "Missing CSRF token validation on state-changing endpoint",
            "SSL certificate validation disabled — MITM vulnerability",
            "User-controlled redirect without whitelist — open redirect vulnerability",
            "Stack trace exposed in error response — leaks internal implementation details",
            "Missing rate limiting on authentication endpoint — brute force risk",
            "Deserialization of untrusted data without validation — RCE risk",
            "Path traversal possible — sanitize file paths before filesystem access",
            "Cookie missing HttpOnly and Secure flags",
            "XML parsing vulnerable to XXE — disable external entity processing",
            "Sensitive data logged in plaintext — PII exposure risk",
            "Missing authorization check — any authenticated user can access admin endpoint",
            "Weak random number generator used for token generation — use secrets module",
            "SQL IN clause built dynamically from user list — injection vector",
            "Password reset token not invalidated after use — token reuse attack",
            "CORS policy too permissive — wildcard origin in production",
            "Missing input length validation — buffer overflow possible",
            "User ID taken from JWT but not verified against database — privilege escalation",
            "Admin endpoint accessible without role check",
            "Temporary file created with predictable name — TOCTOU race condition",
            "Server-side request forgery: user controls URL passed to HTTP client",
            "Missing expiration on session token — session never invalidates",
            "Recursive deserialization of user input — denial of service risk",
            "Object reference exposed directly in URL — IDOR vulnerability",
            "Missing nonce on inline script — Content Security Policy bypass",
            "Database error message returned to client — schema enumeration risk",
            "Regex without anchors — ReDoS possible with crafted input",
            "MD5 used for password hashing — collision attacks feasible",
            "Private key committed to repository — immediate rotation required",
            "HTTP used instead of HTTPS for internal service call",
            "User-supplied filename used in file path — directory traversal",
            "Missing output encoding in template — XSS vulnerability",
            "Session token in URL query parameter — logged and leaked",
            "Account enumeration via different error messages for valid vs invalid user",
            "Multi-factor authentication bypass: fallback SMS not rate-limited",
            "Dependency with known CVE — update to patched version",
            "Encryption key derived from user password without salt — rainbow table attack",
            "Insecure direct object reference on file download endpoint",
            "Missing same-site cookie attribute — CSRF vector",
            "GraphQL introspection enabled in production — schema exposure",
            "User role stored in cookie without signature — tamperable",
            "Missing HSTS header — SSL stripping attack possible",
            "ZIP extraction without size check — zip bomb attack",
            "eval() called on user-supplied string — code injection",
            "Unchecked redirect after login — phishing vector",
            "Email header injection via user-controlled from address",
        ],
        "architecture": [
            "This class has 12 responsibilities — extract domain logic into separate service classes",
            "Circular dependency between UserService and OrderService — introduce interface or event",
            "Business logic in controller layer — move to domain service",
            "Direct database access from UI layer — add repository abstraction",
            "New feature duplicates 80% of existing PaymentProcessor — extract shared base class",
            "Hard-coded dependency on concrete class — inject interface for testability",
            "God object with 40 methods — split by bounded context",
            "Layer skip: presentation calling repository directly without service layer",
            "Event handler contains complex business logic — delegate to domain service",
            "Monolithic transaction spanning 5 aggregates — split into saga or eventual consistency",
            "Tight coupling via shared mutable state — use message passing instead",
            "Temporal coupling: method B must be called before method A — make explicit in API",
            "Domain model polluted with infrastructure concerns — apply hexagonal architecture",
            "Missing abstraction: switch statement will grow with every new payment type — use strategy pattern",
            "Service class depends on 8 other services — too many responsibilities",
            "Violates open/closed principle — adding new type requires modifying existing switch",
            "Infrastructure leaking into domain: JPA annotation in domain entity",
            "Missing anti-corruption layer for external API — domain model will couple to third-party schema",
            "Feature envy: method uses 6 fields from another class — move method",
            "Deep inheritance hierarchy — prefer composition over inheritance",
            "Repository returning domain objects with lazy-loaded collections — N+1 query problem",
            "Service method doing both read and write in same transaction — split queries",
            "Application configuration scattered across 5 classes — centralize in config object",
            "Missing factory for complex object construction — constructor has 12 parameters",
            "Aggregate boundary violation: Order directly accessing Product internals",
            "Domain event not published on state change — other bounded contexts won't know",
            "Anemic domain model: all logic in service, entities are data bags",
            "Missing value object: money represented as plain float",
            "Saga coordinator missing — distributed transaction has no rollback",
            "Synchronous call to slow external service inside domain event handler",
            "Shared database between two microservices — coupling at data layer",
            "Missing interface on external dependency — cannot mock in tests",
            "Port and adapter inverted: domain calls infrastructure directly",
            "Command handler also dispatching events and updating read model — too many concerns",
            "Bounded context importing entity from another bounded context directly",
            "Missing query object — repository has 15 find methods for different filters",
            "Entity exposes setters for all fields — invariants cannot be enforced",
            "Application service orchestrating too many domain services — extract use case",
            "Cross-cutting concern (audit) implemented in each service — use aspect or decorator",
            "Missing specification pattern — complex filtering logic duplicated in 3 places",
            "Domain service doing I/O — push to application layer",
            "Command returns domain entity — should return ID or void",
            "Infrastructure adapter implementing domain logic — violates hexagonal principle",
            "Missing domain event versioning — consumers will break on schema change",
            "Aggregate root allowing direct access to child entities — encapsulation broken",
            "Repository interface in infrastructure layer — should be in domain",
            "Use case class handling multiple user stories — split by responsibility",
            "Missing outbox pattern for reliable event publishing",
            "Read model updated synchronously in command handler — performance bottleneck",
            "Module dependency cycle in monorepo — introduce abstraction layer",
        ],
        "observability": [
            "No logging on exception path — impossible to diagnose failures in production",
            "Missing correlation ID propagation — distributed traces will be disconnected",
            "Database query timing not instrumented — query performance invisible",
            "External API call with no timeout or metric — latency spikes invisible",
            "Error swallowed silently — add structured log with error context",
            "Missing health check endpoint for Kubernetes liveness probe",
            "Cache hit/miss ratio not tracked — cannot optimize cache strategy",
            "Queue depth not monitored — consumer lag invisible until outage",
            "No SLO metric for this critical path — cannot set alerting threshold",
            "Exception caught but only message logged — include stack trace for debugging",
            "Async job has no completion/failure tracking — silent failures possible",
            "Missing structured logging — cannot query logs by user ID or request ID",
            "No metric for retry count — cannot detect cascading failures early",
            "Circuit breaker state not exposed as metric — state changes invisible",
            "Database connection pool exhaustion not alerted — first symptom is timeout errors",
            "Missing span annotations — trace shows duration but not which step is slow",
            "Log level hardcoded to DEBUG in production code — log flooding risk",
            "Business event not published to event stream — audit trail missing",
            "No deadletter queue monitoring — failed messages accumulate silently",
            "Performance-critical loop has zero instrumentation",
            "Missing request duration histogram — cannot set latency SLO",
            "Error rate not tracked per endpoint — cannot detect partial degradation",
            "Kafka consumer lag metric missing — cannot alert before queue grows critical",
            "Missing readiness probe — traffic routed to unready pod",
            "Log messages not including service version — cannot correlate with deployment",
            "Trace context not propagated across async boundary — trace breaks at queue",
            "Missing metric for cache eviction rate — cannot detect memory pressure",
            "No alert on authentication failure spike — brute force invisible",
            "Slow query log not enabled — cannot identify expensive queries",
            "Missing business metric: conversion rate not tracked",
            "Log rotation not configured — disk fills and app crashes silently",
            "Error budget burn rate not calculated — SLO violation not detected early",
            "Missing pod restart counter metric — OOM kills invisible",
            "Trace sampling set to 0.1% — rare errors never captured",
            "Log format not structured JSON — cannot query by field in Elasticsearch",
            "Missing custom metric for queue processing time by partition",
            "Outgoing webhook call not instrumented — third-party latency invisible",
            "Missing deployment marker in metrics — cannot correlate spike with release",
            "No alert on gradual memory leak — only detected after OOM kill",
            "Missing metric for feature flag evaluation — cannot measure rollout impact",
            "Audit log not tamper-evident — compliance requirement not met",
            "Missing distributed lock acquisition metric — contention invisible",
            "Thread pool exhaustion not tracked — manifests as mysterious timeouts",
            "JVM GC pause time not monitored — stop-the-world events invisible",
            "Missing metric for background job duration — SLA breach undetectable",
            "Log sampling without flagging sampled entries — gaps in audit trail",
            "No telemetry on configuration reload — cannot correlate behavior change",
            "Missing error rate per customer tier — cannot detect tier-specific degradation",
            "Kafka producer ack timeout not tracked — data loss invisible",
            "No alert on certificate expiry — outage without warning",
        ],
        "style": [
            "Variable name 'x' is ambiguous in this context — use 'user_count' or similar",
            "Method does two things according to its name and body — split or rename",
            "Magic number 86400 — extract as constant SECONDS_PER_DAY",
            "Inconsistent naming: some methods camelCase, others snake_case in same file",
            "Commented-out code left in — remove or track in issue tracker",
            "Function parameter order inconsistent with rest of codebase convention",
            "Unused import — remove to keep dependencies explicit",
            "Deeply nested if/else — extract early returns for readability",
            "Boolean parameter makes caller unclear — use named parameter or separate methods",
            "Long method with 80 lines — extract logical steps into named helper methods",
            "Missing docstring on public API method",
            "Duplicate condition checked twice in same block",
            "Misleading variable name: 'isValid' returns error message string",
            "Unnecessary else after return statement",
            "Redundant null check on value that was just set",
            "Inconsistent error message format compared to rest of API",
            "Method returns different types depending on input — make return type explicit",
            "String comparison using == instead of .equals() in Java",
            "Hard-coded string duplicated 5 times — extract as constant",
            "Test method name doesn't describe the scenario being tested",
            "Variable shadowing outer scope variable — confusing to readers",
            "Too many parameters: method has 9 arguments — introduce parameter object",
            "Inconsistent abbreviation: 'usr' in some places, 'user' in others",
            "Empty catch block with no explanation — at minimum add a comment",
            "Method name starts with 'get' but has side effects",
            "Single-letter loop variable 'i' in nested loop — use descriptive name",
            "Trailing whitespace on multiple lines",
            "Class name doesn't reflect what it does — rename to clarify intent",
            "Overly defensive null checks that can never trigger",
            "Long boolean expression — extract to well-named predicate method",
            "Ternary expression too long to read on one line — use if/else",
            "Test without assertion — passes trivially",
            "Magic string used as map key in 4 places — extract as constant",
            "Public method that is only called from within the class — make private",
            "Inconsistent level of abstraction within single method",
            "TODO comment without ticket reference — will be forgotten",
            "Dead code path unreachable after refactor — remove",
            "Inconsistent return type naming: some methods 'isX', others 'hasX', others 'checkX'",
            "Method that always returns the same value — likely a stub left in",
            "Type in comment doesn't match parameter type — comment is stale",
            "Overloaded method with too many variants — hard to know which to call",
            "Inconsistent exception type: similar errors throw different exception classes",
            "Variable assigned but never read — dead assignment",
            "Negated condition in if/else — invert for readability",
            "Collection initialized to empty then immediately replaced — remove initialization",
            "toString() method missing from data class — debugging will be painful",
            "Test helper method doing too much — split into focused helpers",
            "Package name doesn't match directory structure",
            "Enum constant names not in UPPER_SNAKE_CASE",
            "Abstract method name implies implementation detail",
        ],
        "false_positive": [
            "This pattern is intentional — the retry logic here handles transient failures by design",
            "The lack of logging is correct here — this is a hot path and logging would impact latency",
            "The 'god class' comment is not applicable — this is a façade intentionally aggregating behavior",
            "SQL string building is safe here — all inputs are enum values, not user-controlled",
            "The circular reference is broken at runtime via lazy loading — no actual cycle in practice",
            "This method is intentionally long to keep the workflow sequential and readable",
            "The magic number is documented in the adjacent comment referencing the RFC",
            "The hardcoded value is a configuration default, overridden by environment variable",
            "Missing test is intentional — this code path is covered by integration test in separate module",
            "The coupling is correct here — these two classes are part of the same aggregate",
            "The exception is swallowed intentionally — failure here is non-critical and recovery is automatic",
            "This appears to be a false positive — the input is validated upstream before reaching this method",
            "The 'duplicate code' is intentional — the two cases are structurally similar but semantically different",
            "No CSRF protection needed here — this is a read-only GET endpoint",
            "The missing abstraction comment is premature optimization — only one implementation exists today",
            "Rate limiting is handled at the API gateway layer — not needed here",
            "The deep nesting is a known trade-off for clarity in this state machine implementation",
            "The password comparison uses constant-time equality — timing attack not applicable",
            "This log level is correct — DEBUG is only enabled via feature flag in prod",
            "The dependency injection is explicit here — no framework needed for this simple case",
            "The static method is appropriate here — it has no state and is a pure utility function",
            "This is not an N+1 — the query is batched by the ORM behind the scenes",
            "The long parameter list is generated by the framework and cannot be changed",
            "Direct field access is intentional — this is a data transfer object, not a domain entity",
            "The commented code is a reference implementation kept for documentation purposes",
            "The coupling to the framework is intentional — this is infrastructure code",
            "Missing error handling is correct — the caller is responsible for handling this exception",
            "The TODO is tracked in JIRA-1234 and scheduled for next sprint",
            "The magic string is a protocol constant defined by an external standard",
            "The deep inheritance is part of the framework contract and cannot be flattened",
            "The synchronous call is correct — this must complete before the response is sent",
            "The missing metric is intentional — this path is not part of the SLO",
            "The hardcoded URL is the canonical service discovery endpoint and is stable",
            "The god object is the entry point of a legacy module being gradually refactored",
            "No abstraction needed — this is the only place this logic is used",
            "The catch-all exception is correct — unhandled errors must not escape this boundary",
            "The unused variable is a placeholder required by the interface contract",
            "The inconsistency is legacy — changing it would break backward compatibility",
            "No docstring needed — this is a private implementation detail",
            "The boolean parameter is clear in context — extracting it would add no clarity",
            "The empty catch block is correct — this exception is expected and ignorable",
            "The static import is standard practice in this test framework",
            "This data class intentionally exposes all fields — it is a public API DTO",
            "The class name is correct — it follows the naming convention of this bounded context",
            "The long method is a migration script and will be deleted after one-time execution",
            "Thread-local storage is correct here — this is intentional per-request state",
            "The metric is tracked at a higher level — duplicating it here would be noise",
            "The abstract method name reflects the domain concept, not the implementation",
            "The singleton is correct here — this resource must be shared across the application",
            "The missing timeout is intentional — this operation must run to completion",
        ],
    }

    result = []
    for label, texts in examples.items():
        for text in texts:
            result.append({"text": text, "label": label})
    return result


def process_hf_samples(samples: list[dict]) -> list[dict]:
    """Map HuggingFace samples to (text, label) pairs."""
    labeled = []
    skipped = 0

    for sample in samples:
        text = build_finding_text(sample)
        if not text:
            skipped += 1
            continue

        label = classify_text(text)
        if label is None:
            skipped += 1
            continue

        labeled.append({"text": text, "label": label})

    print(f"  Mapeados: {len(labeled)}, Ignorados: {skipped}")
    return labeled


def balance_and_split(examples: list[dict], target_per_class: int = 400) -> dict[str, list[dict]]:
    """Balance classes and create stratified 80/10/10 splits."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_label[ex["label"]].append(ex)

    print("\nDistribuição antes do balanceamento:")
    for label, items in sorted(by_label.items()):
        print(f"  {label}: {len(items)}")

    train_all, val_all, test_all = [], [], []

    for label in sorted(by_label.keys()):
        items = by_label[label]
        random.shuffle(items)

        available = len(items)
        if available >= target_per_class:
            items = items[:target_per_class]
        else:
            print(f"  AVISO: {label} tem apenas {available} exemplos (alvo: {target_per_class})")

        n = len(items)
        n_val = max(1, int(n * 0.10))
        n_test = max(1, int(n * 0.10))
        n_train = n - n_val - n_test

        train_all.extend(items[:n_train])
        val_all.extend(items[n_train:n_train + n_val])
        test_all.extend(items[n_train + n_val:])

    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    print(f"\nSplits finais: train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")
    print("\nDistribuição no train:")
    for label, count in sorted(Counter(e["label"] for e in train_all).items()):
        print(f"  {label}: {count}")

    return {"train": train_all, "val": val_all, "test": test_all}


def save_splits(splits: dict[str, list[dict]]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, examples in splits.items():
        path = SPLITS_DIR / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Salvo: {path} ({len(examples)} exemplos)")


def main():
    random.seed(42)

    raw_samples = load_hf_dataset()

    if raw_samples and "label" in raw_samples[0]:
        print(f"\nDataset já tem labels — usando diretamente ({len(raw_samples)} exemplos)")
        labeled = raw_samples
    else:
        print(f"\nMapeando {len(raw_samples)} exemplos para labels...")
        labeled = process_hf_samples(raw_samples)

    if len(labeled) < 100:
        print("Dataset muito pequeno após mapeamento. Complementando com sintético...")
        labeled.extend(generate_synthetic_dataset())

    splits = balance_and_split(labeled, target_per_class=400)
    save_splits(splits)
    print("\nDataset pronto.")


if __name__ == "__main__":
    main()
