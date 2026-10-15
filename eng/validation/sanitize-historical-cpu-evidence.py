#!/usr/bin/env python3
"""Redact an obsolete CPU evidence process snapshot without displaying values.

Restricted to generated JSON evidence below explicitly supplied directories.
Numerical/timing values remain intact. Only the historical field and SHA256
references to edited evidence files change; process arguments are never read
from the live system or printed.
"""
import argparse
import hashlib
import json
from pathlib import Path

FIELD = "pre_timing_processes"
REDACTION = "Historical raw command-line capture removed; it included unrelated service credential arguments. No numerical or timing results changed."


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    files = set()
    for root in args.evidence_root:
        resolved = root.resolve()
        relative = resolved.relative_to(workspace)
        if relative.parts[0] != "artifacts" and relative.parts[:2] != ("docs", "validation"):
            raise ValueError("Only ignored generated evidence directories are allowed")
        files.update(resolved.rglob("*.json"))
    documents = {}
    for path in files:
        try:
            value = json.loads(path.read_text())
        except (UnicodeError, ValueError):
            continue
        documents[path] = value
    changed_hashes, redacted, checksum_updates = {}, [], set()

    def save(path, value):
        previous = sha(path)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        changed_hashes[previous] = sha(path)

    for path, value in documents.items():
        if isinstance(value, dict) and FIELD in value and value[FIELD] != REDACTION:
            value[FIELD] = REDACTION
            save(path, value)
            redacted.append(str(path.relative_to(workspace)))

    def update(value, hashes):
        changed = False
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str) and key.endswith("sha256") and child in hashes:
                    value[key] = hashes[child]
                    changed = True
                elif key == "report_hashes" and isinstance(child, dict):
                    for name, digest in child.items():
                        if isinstance(digest, str) and digest in hashes:
                            child[name] = hashes[digest]
                            changed = True
                else:
                    changed |= update(child, hashes)
        elif isinstance(value, list):
            for child in value:
                changed |= update(child, hashes)
        return changed

    # References can form an acyclic report chain (runtime -> summary -> index).
    # Never hide a cycle behind an infinite rewrite loop.
    for _ in range(len(documents) + 1):
        updates = 0
        snapshot = dict(changed_hashes)
        for path, value in documents.items():
            if update(value, snapshot):
                save(path, value)
                checksum_updates.add(str(path.relative_to(workspace)))
                updates += 1
        if not updates:
            break
    else:
        raise ValueError("Cyclic evidence checksum references require manual reconciliation")
    print(json.dumps(dict(redacted_count=len(redacted), redacted_paths=sorted(redacted),
                          checksum_update_count=len(checksum_updates), checksum_update_paths=sorted(checksum_updates)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
