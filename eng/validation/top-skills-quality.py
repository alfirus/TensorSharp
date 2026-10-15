#!/usr/bin/env python3
"""Live local model A/B probe through the real OpenAI endpoint and artifact routes.

Requires a prestarted TensorSharp server with the shared skill bundle and code-exec.
Scores task-specific behavior and downloads actual saved artifacts. Regex scores are
limited behavioral checks; retained responses require review, not a quality rating.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base: str, path: str, payload: dict | None = None, timeout: int = 600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        # Decode known API routes as JSON; artifact routes return literal bytes.
        return json.loads(body) if path.startswith("/v1/") or path == "/api/skills" else body


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executed_tool_outputs(section: str) -> str:
    """Decode tool content from host JSON logs instead of matching escaped newlines."""
    outputs = []
    for line in section.splitlines():
        if " fullInput=" not in line:
            continue
        try:
            messages = json.loads(line.split(" fullInput=", 1)[1])
        except ValueError:
            continue
        outputs.extend(message.get("content", "") for message in messages if message.get("role") == "tool")
    return "\n".join(outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18592")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, help="Host log for independent proof of executed red/green commands")
    parser.add_argument("--tasks", type=Path, default=Path("InferenceWeb.Tests/Fixtures/TopSkills/tasks.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skills", nargs="*")
    parser.add_argument("--selected-only", action="store_true", help="Recheck final adapted manifests without repeating the baseline arm")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-skill-rounds", type=int, default=24, help="Record the prestarted server's actual round cap")
    parser.add_argument("--context-tokens", type=int, default=16384)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = request(args.base, "/v1/models")["data"][0]["id"]
    catalog = request(args.base, "/api/skills")
    tasks = json.loads(args.tasks.read_text())
    available = {skill["id"] for skill in catalog["skills"]}
    manifests = {task["skill"]: hashlib.sha256(request(args.base, "/api/skills/" + task["skill"] + "/files/SKILL.md")).hexdigest()
        for task in tasks if task["skill"] in available}
    unavailable = [task["skill"] for task in tasks if task["skill"] not in available]
    if not args.skills:
        tasks = [task for task in tasks if task["skill"] in available]
    if unavailable:
        print("Not run (absent from active catalog; not passes): " + ", ".join(unavailable), flush=True)
    model_hash = sha256(args.model_file)
    binary_hash = sha256(args.binary)
    modules = {name: sha256(args.binary.parent / name) for name in
        ["TensorSharp.Server.dll", "TensorSharp.Chat.dll", "TensorSharp.AgentHost.dll", "TensorSharp.Runtime.dll",
         "TensorSharp.Models.dll", "TensorSharp.Backends.GGML.dll", "libGgmlOps.dylib"] if (args.binary.parent / name).is_file()}
    rows = []
    for task in tasks:
        skill = task["skill"]
        if args.skills and skill not in args.skills:
            continue
        for enabled in ((True,) if args.selected_only else (False, True)):
            label = skill + ("-skill" if enabled else "-baseline")
            payload = {"model": model, "messages": [{"role": "user", "content": task["task"]}], "skills": [skill] if enabled else [], "skills_discovery": False, "stream": False, "temperature": 0, "seed": 42, "think": False, "max_tokens": args.max_tokens}
            start = time.perf_counter()
            try:
                response = request(args.base, "/v1/chat/completions", payload)
                elapsed = time.perf_counter() - start
                (args.output / (label + ".json")).write_text(json.dumps(response, indent=2))
                content = response["choices"][0]["message"].get("content") or ""
                artifacts = {}
                for path in set(re.findall(r"/api/code/artifacts/[^\s)\]<>\"'`\\]+", content)):
                    try:
                        data = request(args.base, path.rstrip(".,;"))
                        if isinstance(data, bytes):
                            name = path.rsplit("/", 1)[-1]
                            artifacts[name] = data.decode("utf-8", errors="replace")
                            (args.output / (label + "-" + name)).write_bytes(data)
                    except (urllib.error.URLError, ValueError) as error:
                        artifacts[path] = "DOWNLOAD ERROR: " + str(error)
                joined = content + "\n" + "\n".join(artifacts.values())
                checks = {"nonempty_answer": bool(content.strip()), "no_unhandled_tools": not response["choices"][0]["message"].get("tool_calls"), "no_raw_tool_markup": not re.search(r"(?:^|\n)\s*call:[\w-]+\{|<\|tool_call\|>", content), "finished_without_length_limit": response["choices"][0]["finish_reason"] != "length"}
                if skill == "grill-me":
                    checks.update(asks_questions="?" in content, discusses_conflicts="conflict" in content.lower(), recommends_answer=bool(re.search(r"recommend|suggest|➡", content, re.I)))
                elif skill == "grill-with-docs":
                    checks.update(saved_glossary="GLOSSARY.md" in artifacts, preserves_reservation="Reservation" in joined and "15" in joined, asks_unsettled_decision="?" in content)
                elif skill == "improve-codebase-architecture":
                    html = artifacts.get("architecture-review.html", "")
                    checks.update(saved_report=bool(html), offline_report=bool(html) and not re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://', html, re.I), discusses_both_files="checkout.py" in joined and "checkout_routes.py" in joined, before_after="before" in joined.lower() and "after" in joined.lower())
                elif skill == "tdd":
                    log = args.server_log.read_text(errors="replace") if args.server_log else ""
                    marker = "/v1/chat/completions request: id=" + response["id"]
                    section = log.split(marker, 1)[-1].split("/v1/chat/completions request: id=", 1)[0] if marker in log else ""
                    executed = executed_tool_outputs(section)
                    red = re.search(r"FAIL: test_", executed)
                    green = re.search(r"Ran [1-9]\d* tests? in [^\n]+\n\s*\nOK\b", executed)
                    checks.update(saved_implementation="clamp_percent.py" in artifacts, saved_tests="test_clamp_percent.py" in artifacts,
                        actual_red_run=bool(red), actual_green_run=bool(green))
                    checks["red_before_green"] = bool(red and green and red.start() < green.start())
                    checks["independent_final_cases"] = False
                    if "clamp_percent.py" in artifacts and "test_clamp_percent.py" in artifacts:
                        final = args.output / (label + "-final")
                        final.mkdir(exist_ok=True)
                        for filename in ("clamp_percent.py", "test_clamp_percent.py"):
                            (final / filename).write_text(artifacts[filename])
                        code = "import sys,unittest;sys.path.insert(0,sys.argv[1]);from clamp_percent import clamp_percent;assert [clamp_percent(x) for x in [-5,0,42,100,150]]==[0,0,42,100,100];result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(sys.argv[1],pattern='test_clamp_percent.py'));sys.exit(0 if result.wasSuccessful() and result.testsRun>0 else 1)"
                        verified = subprocess.run([sys.executable, "-I", "-c", code, str(final.resolve())], capture_output=True, text=True, timeout=20)
                        (final / "independent-verification.txt").write_text(verified.stdout + verified.stderr)
                        checks["independent_final_cases"] = verified.returncode == 0
                elif skill == "handoff":
                    doc = artifacts.get("handoff.md", "")
                    checks.update(saved_handoff=bool(doc), credential_redacted="CEDAR_SECRET_VALUE_7392" not in joined, records_unrun_tests="not" in doc.lower() and "run" in doc.lower(), records_blocker="undecided" in doc.lower())
                elif skill == "frontend-design":
                    html = artifacts.get("cedar-landing.html", "")
                    checks.update(saved_landing_page=bool(html), complete_html=bool(re.search(r"<html\b", html, re.I)) and bool(re.search(r"</html>", html, re.I)),
                        offline=bool(html) and not re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://', html, re.I),
                        required_content=all(term in html for term in ["Cedar Notes", "Your notes stay yours", "Offline first", "Encrypted sync", "Manual conflict review", "Get started"]),
                        internal_cta=bool(re.search(r'href=["\']#start["\']', html)) and bool(re.search(r'id=["\']start["\']', html)))
                rows.append({"skill": skill, "enabled": enabled, "elapsed_seconds": elapsed, "usage": response.get("usage"), "checks": checks, "all_checks_pass": all(checks.values()), "artifact_names": list(artifacts)})
            except (urllib.error.URLError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
                rows.append({"skill": skill, "enabled": enabled, "elapsed_seconds": time.perf_counter() - start, "all_checks_pass": False, "error": str(error)})
            (args.output / "report.json").write_text(json.dumps({"model": model, "backend": "ggml_cpu", "model_sha256": model_hash, "binary_sha256": binary_hash, "module_sha256": modules, "fixture_sha256": sha256(args.tasks), "skill_manifest_sha256": manifests, "catalog_count": len(catalog["skills"]), "unavailable_tasks_not_run": unavailable, "context_tokens": args.context_tokens, "max_skill_rounds": args.max_skill_rounds, "max_tokens_per_round": args.max_tokens, "temperature": 0, "seed": 42, "limitations": ["One task per skill, one greedy sample per arm; interview tasks cover their first turn only.", "Elapsed request time includes tool rounds and prompt/cache behavior; this is not a controlled tokens/second kernel benchmark.", "Checks are task-specific behavioral oracles; inspect retained responses and artifacts before concluding overall quality.", "No phone/device execution or external browser/renderer capability is measured by this script."], "rows": rows}, indent=2) + "\n")
            print(label + ": " + json.dumps(rows[-1]), flush=True)
    enabled_rows = [row for row in rows if row["enabled"]]
    return 0 if enabled_rows and all(row["all_checks_pass"] for row in enabled_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
