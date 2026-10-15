#!/usr/bin/env python3
"""Adapt the audited, pinned skills.sh candidates to TensorSharp's tool contract.

Download candidates with skill-installer into an ignored evidence directory first.
This tool treats their Markdown as data, never executes upstream code, and produces
self-contained skill folders: explicit selection must work with discovery disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

REVISION = "49dd158d1076134a641b33efb035946536778336"
DESCRIPTIONS = {
    "grill-me": "Interview the user to stress-test a plan or decision, one independent set of questions at a time.",
    "grill-with-docs": "Interview a plan, record agreed terms and decisions, and return saved document links before questions.",
    "improve-codebase-architecture": "Review a supplied codebase for simpler interfaces and deeper modules; produce an offline HTML report.",
    "tdd": "Build features or fix bugs test-first through agreed public interfaces using the red-green loop.",
    "handoff": "Write a concise handoff from the conversation so another agent can continue; redact sensitive values.",
}
PATHS = {
    "grill-me": "skills/productivity/grill-me",
    "grill-with-docs": "skills/engineering/grill-with-docs",
    "improve-codebase-architecture": "skills/engineering/improve-codebase-architecture",
    "tdd": "skills/engineering/tdd",
    "handoff": "skills/productivity/handoff",
}


def body(path: Path) -> str:
    return path.read_text().split("---", 2)[2].strip() + "\n"


def read_reference(skill: str, filename: str) -> str:
    return f'`skills_read` with `skill="{skill}"`, `path="{filename}"`'


def portable(text: str) -> str:
    text = text.replace("dispatch a sub-agent to find it", "use the available file tools to find it (delegate only if an agent tool is offered)")
    text = text.replace("Then spawn a sub-agent to walk the codebase.", "Walk the supplied codebase using the available file tools; delegate only if an agent tool is offered.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, action="append", help="Apply live quality reports in chronological order (repeatable)")
    args = parser.parse_args()
    candidates, dest = args.candidates, args.dest
    prior_path = dest / "top-skills-provenance.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    rejected = {entry["name"]: entry for entry in prior.get("rejected_after_validation", [])}
    earlier = list(prior.get("earlier_validation_failures", []))
    validated = {entry["name"]: entry.get("workflow_validation") or entry["model_capacity_requirement"]
        for entry in prior.get("installed", []) if entry.get("workflow_validation") or entry.get("model_capacity_requirement")}
    for report_path in args.quality_report or []:
        report = json.loads(report_path.read_text())
        for row in report["rows"]:
            if row.get("oracle_correction"):
                for entry in earlier:
                    if entry["name"] == row["skill"] and entry.get("model_sha256") == report["model_sha256"]:
                        entry["failed_checks"] = [key for key, passed in row.get("checks", {}).items() if not passed]
                        entry["oracle_correction"] = row["oracle_correction"]
            if row["enabled"] and row["all_checks_pass"]:
                validated[row["skill"]] = {"validated_model": report["model"], "model_sha256": report["model_sha256"],
                    "context_tokens": report.get("context_tokens", 16384), "max_skill_rounds": report.get("max_skill_rounds", 8),
                    "max_tokens_per_round": report.get("max_tokens_per_round")}
            if row["enabled"] and row["all_checks_pass"] and row["skill"] in rejected:
                previous = rejected.pop(row["skill"])
                previous["subsequent_passing_model"] = report["model"]
                previous["subsequent_passing_model_sha256"] = report["model_sha256"]
                earlier.append(previous)
            if row["enabled"] and not row["all_checks_pass"]:
                validated.pop(row["skill"], None)
                if row["skill"] in rejected and rejected[row["skill"]].get("model_sha256") != report["model_sha256"]:
                    previous = rejected[row["skill"]].copy()
                    previous["subsequent_failed_model"] = report["model"]
                    earlier.append(previous)
                rejected[row["skill"]] = {"name": row["skill"], "model": report["model"], "model_sha256": report["model_sha256"], "backend": report["backend"],
                    "context_tokens": report.get("context_tokens"), "max_skill_rounds": report.get("max_skill_rounds"), "max_tokens_per_round": report.get("max_tokens_per_round"),
                    "skill_manifest_sha256": report.get("skill_manifest_sha256", {}).get(row["skill"]),
                    "failed_checks": [key for key, passed in row.get("checks", {}).items() if not passed], "error": row.get("error"), "reason": "Failed the real local model workflow after tool-contract adaptations and targeted fixes; removed from active app bundles."}
    # Keep a single history entry for each actual failed check set. Later passes
    # add their tested model to that entry without discarding earlier evidence.
    unique = {}
    for entry in earlier:
        key = (entry["name"], entry.get("model_sha256"), entry.get("max_skill_rounds"), entry.get("max_tokens_per_round"),
            entry.get("skill_manifest_sha256"), tuple(entry.get("failed_checks", [])))
        previous = unique.get(key, {})
        previous.update(entry)
        unique[key] = previous
    earlier = list(unique.values())
    license_text = (candidates / "mattpocock-LICENSE.txt").read_text()
    records = []
    for name, description in DESCRIPTIONS.items():
        target = dest / name
        target.mkdir(parents=True, exist_ok=True)
        # Codex-specific invocation descriptors name tools TensorSharp does not expose.
        shutil.rmtree(target / "agents", ignore_errors=True)
        source = candidates / name
        imported = [(name, "SKILL.md")]

        def reference(dependency: str, filename: str = "SKILL.md", rename: str | None = None) -> None:
            refname = rename or filename
            text = body(candidates / dependency / filename) if filename == "SKILL.md" else (candidates / dependency / filename).read_text()
            if dependency == "codebase-design" and filename != "SKILL.md":
                text = text.replace("[SKILL.md](SKILL.md)", "[codebase-design.md](codebase-design.md)")
            if filename == "DESIGN-IT-TWICE.md":
                text = "Use parallel agents only when an agent tool is actually offered. Otherwise explore the alternative designs sequentially with the available file tools.\n\n" + text
            (target / refname).write_text(portable(text))
            imported.append((dependency, filename))

        text = body(source / "SKILL.md")
        if name in {"grill-me", "grill-with-docs"}:
            text = "# Plan interview\n\n" + portable(body(candidates / "grilling" / "SKILL.md"))
            imported.append(("grilling", "SKILL.md"))
            if name == "grill-with-docs":
                text = "# Save decisions and interview\n\nWhen the user requests a glossary or decision document, save the already-agreed facts first. Immediately copy the exact Markdown artifact link from the successful file-tool result as the first line of your answer; then ask the unresolved frontier questions. A filename or a statement that the document was saved does not deliver it. Keep agreed facts separate from recommendations.\n\n" + text
                text += "\nAs terminology and durable decisions are resolved, read " + read_reference(name, "domain-modeling.md") + ". Use that guidance to maintain the glossary and ADRs in the session workspace.\n"
                reference("domain-modeling", rename="domain-modeling.md")
                for filename in ("GLOSSARY-FORMAT.md", "ADR-FORMAT.md"):
                    reference("domain-modeling", filename)
        elif name == "improve-codebase-architecture":
            text = text.replace('Call the Skill tool with "codebase-design"', "Read " + read_reference(name, "codebase-design.md"))
            text = text.replace('call the Skill tool with "grilling"', "read " + read_reference(name, "grilling.md"))
            text = text.replace('call the Skill tool with "domain-modeling"', "read " + read_reference(name, "domain-modeling.md"))
            text = text.replace('Call the Skill tool with "codebase-design"', "Read " + read_reference(name, "codebase-design.md"))
            text = re.sub(r"Write a self-contained HTML file.*?and tell them the absolute path\.", "Write a self-contained `architecture-review.html` in the session workspace with the available file tool. Return its artifact link. If no workspace/file tools are offered, provide the report in the answer and state that no file was saved.", text, flags=re.S)
            text = re.sub(r"The report uses \*\*Tailwind via CDN\*\*.*?Be visual\.", "Use inline CSS and inline SVG/HTML diagrams. The report must work offline, with no external scripts, stylesheets, fonts, or Mermaid runtime.", text, flags=re.S)
            for dependency in ("codebase-design", "grilling", "domain-modeling"):
                reference(dependency, rename=dependency + ".md")
            for dependency, filenames in {"codebase-design": ("DEEPENING.md", "DESIGN-IT-TWICE.md"), "domain-modeling": ("GLOSSARY-FORMAT.md", "ADR-FORMAT.md")}.items():
                for filename in filenames:
                    reference(dependency, filename)
            (target / "HTML-REPORT.md").write_text("""# Offline architecture report

Create a single HTML document with inline CSS and SVG. No external resources.
Include a header naming the supplied codebase, a legend, and one article per
candidate with file paths, problem, proposed solution, benefits, before/after
diagrams, recommendation strength, and relevant ADR conflicts. Finish with a top
recommendation. Give diagrams text labels and keep the layout readable on phones.
Use the domain glossary and the bundled codebase-design vocabulary. Save through
the host's workspace file tool and return the artifact link.
""")
            imported.append((name, "HTML-REPORT.md"))
        elif name == "tdd":
            text += "\n## Execute and prove each cycle\n\nFor an approved interface, first create one public test and the minimal failing implementation (for example a stub). Run that test with the offered shell tool and confirm its real failure before implementing the behavior. A test name containing 'red' is not evidence of a red run. Implement only that slice and run it again to prove green; then repeat for the remaining agreed behavior. Report actual failing and passing command results, never inferred outcomes or runs you did not perform.\n"
            text = text.replace('call the Skill tool with "codebase-design"', "read " + read_reference(name, "codebase-design.md"))
            text = text.replace("see the `code-review` skill", "review the agreed public interface and test results with the user")
            reference("codebase-design", rename="codebase-design.md")
            for filename in ("DEEPENING.md", "DESIGN-IT-TWICE.md"):
                reference("codebase-design", filename)
            for filename in ("tests.md", "mocking.md"):
                reference(name, filename)
        elif name == "handoff":
            text += "\nReplace secret values with `[REDACTED]` everywhere, including the saved file, response, annotations, examples, and any explanation of redaction. Never repeat an original secret in a sentence saying it was redacted. Preserve only the fact that a credential was removed.\n"
            text = text.replace("Save to the temporary directory of the user's OS (`$TMPDIR`, else `/tmp`; `%TEMP%` on Windows) - not the current workspace.", "Save `handoff.md` in the session workspace using the offered file tool and return its artifact link. If no file tool/workspace is available, provide the full handoff in the answer and state that no file was saved.")
            text = text.replace("naming which skills the next agent should call the Skill tool for", "naming only relevant skills actually listed in the available catalog (the next agent can read them with `skills_read`)")
        text = portable(text)
        text += "\n## Host capabilities\n\nUse only tools offered in this conversation. Read supplied workspace files with `read_file`; create artifacts with `write_file` and edit existing files with `apply_patch` when offered. Do not invent a `Skill` tool. After creating any file, include its exact Markdown download link from the tool result in your answer, even when the rest of your answer asks the user a decision question. Never replace that link with a filename or a claim that the file exists. If a required file, workspace, executable, or tool is unavailable, explain the limitation and do not claim it ran. A user's already-agreed scope and test interface count as agreement; do not ask the same question again. Apply the workflow directly after reading it; keep progress narration brief.\n"
        capacity = next((entry for entry in earlier if entry["name"] == name and "E2B" in entry.get("model", "")), None) if name in validated else None
        compatibility = "TensorSharp or TensorAgent; workspace file tools required for saved artifacts and code changes."
        if capacity:
            description += " Tested with Gemma4 E4B at 16K context and 24 tool rounds."
            compatibility += " Tested on Gemma4 E4B Q8_0 with 16384-token context and up to 24 tool rounds. E2B failed the earlier eight-round task; E2B at 24 rounds and other models are not validated."
        manifest = f"---\nname: {name}\ndescription: {description}\nlicense: MIT\ncompatibility: {compatibility}\nmetadata:\n  source: mattpocock/skills\n  revision: {REVISION}\n---\n\n{text}"
        (target / "SKILL.md").write_text(manifest)
        (target / "LICENSE.txt").write_text(license_text)
        record = {"name": name, "repository": "mattpocock/skills", "revision": REVISION, "path": PATHS[name], "license": "MIT", "adaptations": "self-contained dependency references, TensorSharp tool names, workspace artifacts, optional delegation, offline architecture report", "upstream_files": [{"skill": dependency, "file": filename, "sha256": hashlib.sha256((candidates / dependency / filename).read_bytes()).hexdigest()} for dependency, filename in imported], "integrated_files": {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(target.rglob("*")) if path.is_file()}}
        if name in validated:
            record["workflow_validation"] = dict(validated[name], scope="One fixed task through the real local HTTP endpoint; not a general model-quality guarantee.")
        if capacity:
            record["model_capacity_requirement"] = dict(validated[name], earlier_failed_model=capacity["model"], earlier_failed_max_skill_rounds=8, earlier_model_at_24_rounds="not validated", other_models="not validated")
        records.append(record)
    audited = [
        {"rank": 1, "name": "find-skills", "repository": "vercel-labs/skills", "revision": "13e4063a1cf913f5606d57d42ab83a86f5001e04", "license": "MIT", "disposition": "excluded", "reason": "Requires npx skills, recommends global installation, and cannot complete discovery/installation in the in-process mobile shell."},
        {"rank": 2, "name": "grill-me", "disposition": "installed"},
        {"rank": 3, "name": "grill-with-docs", "disposition": "installed"},
        {"rank": 4, "name": "agent-browser", "repository": "vercel-labs/agent-browser", "revision": "d9570915f4dd6504dee4070379d2c4f79d81dbb2", "license": "Apache-2.0", "disposition": "excluded", "reason": "Only a discovery stub; requires an external native CLI and Chrome/CDP, unavailable in TensorAgent's in-process mobile runtime. Existing validated desktop playwright remains available."},
        {"rank": 5, "name": "improve-codebase-architecture", "disposition": "installed"},
        {"rank": 6, "name": "tdd", "disposition": "installed"},
        {"rank": 7, "name": "setup-matt-pocock-skills", "repository": "mattpocock/skills", "revision": REVISION, "license": "MIT", "disposition": "excluded", "reason": "Assumes a persistent repository, git remote/config and optional gh/glab tracker mutation. No such repository/tracker integration is established in a TensorAgent session; not validated end to end."},
        {"rank": 8, "name": "frontend-design", "repository": "anthropics/skills", "audited_revision": "dbd4588f9e1033efb41dad4bef2f7947c8993d44", "license": "Apache-2.0", "disposition": "preexisting retained", "reason": "Retain the established design body with a targeted workspace-delivery hint; no duplicate upstream pack is installed."},
        {"rank": 9, "name": "handoff", "disposition": "installed"},
        {"rank": 10, "name": "hyperframes-cli", "repository": "heygen-com/hyperframes", "revision": "e8eb358fe21d6f1a611e04f8df27ca366b337ce9", "license": "Apache-2.0", "disposition": "excluded", "reason": "Requires Node >=22, npx, FFmpeg and browser rendering plus multiple missing sibling skills. No mobile renderer or local end-to-end renderer validation is available."},
    ]
    for entry in audited:
        if entry["name"] == "frontend-design":
            existing = dest / entry["name"]
            entry["integrated_files"] = {str(path.relative_to(existing)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(existing.rglob("*")) if path.is_file()}
            if entry["name"] in validated:
                entry["workflow_validation"] = dict(validated[entry["name"]], scope="One fixed offline HTML task; browser render coverage is recorded separately.")
        if entry["name"] in rejected:
            entry["disposition"] = "excluded after live validation"
            entry["reason"] = rejected[entry["name"]]["reason"]
    active = []
    for record in records:
        if record["name"] in rejected:
            rejected[record["name"]]["source_provenance"] = record
            shutil.rmtree(dest / record["name"])
        else:
            active.append(record)
    (dest / "top-skills-provenance.json").write_text(json.dumps({"leaderboard": "https://www.skills.sh/", "snapshot_date": "2026-10-13", "top_ten_audit": audited, "installed": active, "rejected_after_validation": list(rejected.values()), "earlier_validation_failures": earlier}, indent=2) + "\n")
    verdict_path = dest / "verdicts.json"
    verdicts = json.loads(verdict_path.read_text())
    verdicts = [entry for entry in verdicts if entry["name"] not in DESCRIPTIONS]
    verdicts.extend({"name": record["name"], "ok": True, "kind": "instructions only", "scripts": 0, "reasons": [], "validation": "portable instructions and bundled references; live model quality is recorded separately"} for record in active)
    verdict_path.write_text(json.dumps(sorted(verdicts, key=lambda entry: entry["name"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
