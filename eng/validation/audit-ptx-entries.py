#!/usr/bin/env python3
"""Compare every PTX entry body; compile inputs with identical nvcc/flags first."""
import argparse
import json
import re
from pathlib import Path


def entries(path):
    source = path.read_text()
    result = {}
    for match in re.finditer(r"(?m)^\.visible \.entry\s+(\w+)\s*\(", source):
        start = source.index("{", match.start())
        depth, end = 1, start + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        body = re.sub(r"//[^\n]*", "", source[match.start():end])
        result[match.group(1)] = re.sub(r"\s+", " ", body).strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    before, after = entries(args.before), entries(args.after)
    report = {"before_entries": len(before), "after_entries": len(after),
              "changed_prior_entries": [name for name, body in before.items() if name in after and after[name] != body],
              "added_entries": sorted(after.keys() - before.keys()),
              "missing_entries": sorted(before.keys() - after.keys())}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
    return int(bool(report["changed_prior_entries"] or report["missing_entries"]))


if __name__ == "__main__":
    raise SystemExit(main())
