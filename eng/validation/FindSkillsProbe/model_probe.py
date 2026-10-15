"""Validate find-skills through a real model and TensorSharp/TensorAgent's shared chat host.

Requires a running tool-capable model, the bundled find-skills skill, an isolated
writable skill install directory, --skills-allow-network, --skills-allow-exec,
and at least 12 tool rounds. Does not substitute canned model responses.
"""
import argparse
import json
import pathlib
import platform
import re
import time
import urllib.request
from html.parser import HTMLParser


def request(endpoint, prompt, max_tokens, cookie=None, temperature=0.0, seed=42):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "maxTokens": max_tokens,
        "newChat": True,
        "multi_agent": False,
        "skills": ["find-skills"],
        "skills_discovery": False,
        "temperature": temperature,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    frames = []
    answer = ""
    first_token = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: {"):
                continue
            frame = json.loads(line[6:])
            frames.append(frame)
            if isinstance(frame.get("token"), str):
                if first_token is None:
                    first_token = time.perf_counter() - started
                answer += frame["token"]
            if isinstance(frame.get("replace"), str):
                answer = frame["replace"]
    return {
        "seconds": time.perf_counter() - started,
        "first_visible_token_seconds": first_token,
        "answer": answer,
        "frames": frames,
    }


def verify(result, expected_tools, expected_skill):
    observed = {frame.get("skill_step") for frame in result["frames"] if frame.get("ok")}
    missing = set(expected_tools) - observed
    if missing:
        raise AssertionError(f"Missing successful model tool calls: {sorted(missing)}; observed {sorted(observed)}")
    if not any(frame.get("done") and not frame.get("error") and not frame.get("aborted") and not frame.get("truncated")
               for frame in result["frames"]):
        raise AssertionError("No successful terminal chat frame")
    for tool in ("skills_install", "skills_read"):
        if not any(frame.get("skill_step") == tool and frame.get("skill") == expected_skill and frame.get("ok")
                   for frame in result["frames"]):
            raise AssertionError(f"No successful {tool} for the requested skill {expected_skill}")
    if "skills_run" in expected_tools and not any(
            frame.get("skill_step") == "skills_run" and frame.get("skill") == expected_skill
            and frame.get("detail") == "scripts/aggregate_benchmark.py" and frame.get("ok")
            for frame in result["frames"]):
        raise AssertionError("The actual requested upstream helper was not run successfully")
    if not result["answer"].strip():
        raise AssertionError("The model returned no visible answer")


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []
        self.viewport = False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.viewport |= tag == "meta" and dict(attrs).get("name", "").lower() == "viewport"

    def handle_data(self, data):
        self.text.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5049")
    parser.add_argument("--connection-file", help="TensorAgentHost connection.json; uses its local endpoint and cookie without recording the cookie")
    parser.add_argument("--out", default="artifacts/find-skills/model-report.json")
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.0, help="Requested sampling temperature (default: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Requested sampler seed (default: 42); -1 allows an unseeded run")
    parser.add_argument("--model", required=True, help="Actual loaded model path/name, recorded for provenance")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--case", choices=("all", "instructions", "script"), default="all")
    args = parser.parse_args()
    cookie = None
    if args.connection_file:
        connection = json.loads(pathlib.Path(args.connection_file).read_text(encoding="utf-8"))
        args.endpoint = connection["baseUrl"]
        cookie = connection["cookie"]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "backend": args.backend,
        "platform": platform.platform(),
        "endpoint": args.endpoint,
        "host": "TensorAgent AgentAppHost" if args.connection_file else "TensorSharp.Server.Host",
        "requested_sampling": {"temperature": args.temperature, "seed": args.seed},
        "limitations": [
            "One local model/backend and one observation per selected case; not a model/device compatibility sweep.",
            "The instruction skill check verifies semantic HTML requirements, not visual rendering or subjective design quality.",
            "The model script case exercises the real upstream helper's help entry point; the managed probe separately checks artifact-producing execution and arithmetic.",
            "This shared chat host exercise does not launch TensorAgent's native GUI.",
        ],
        "passed": False,
    }
    try:
        if args.case in ("all", "instructions"):
            result = request(args.endpoint,
                "Use find-skills to find the official frontend-design skill from anthropics/skills. "
                "Search, inspect it, install it using the inspected revision, and read its SKILL.md. "
                "I authorize that installation. Then apply its design guidance to write a complete, "
                "small standalone HTML page for a neighborhood night market named Moonlit Market. "
                "Include a viewport meta tag, responsive inline CSS, a main landmark, the exact title "
                "Moonlit Market, Saturday 6–10 PM, and a button labeled View stalls. "
                "Return the HTML in one code block. Keep it below 100 lines.", args.max_tokens, cookie, args.temperature, args.seed)
            report["instructions"] = result
            verify(result, ("skills_search", "skills_inspect", "skills_install", "skills_read"), "frontend-design")
            blocks = re.findall(r"```html\s*(.*?)```", result["answer"], re.DOTALL | re.IGNORECASE)
            if len(blocks) != 1:
                raise AssertionError("Expected one complete HTML code block")
            html = blocks[0]
            page = PageParser()
            page.feed(html)
            if not page.viewport or not {"html", "head", "body", "main", "style", "button"}.issubset(page.tags):
                raise AssertionError("Generated HTML is missing required semantic elements")
            visible = " ".join(page.text).lower()
            for expected in ("moonlit market", "view stalls", "saturday"):
                if expected not in visible:
                    raise AssertionError(f"Generated HTML missing {expected!r}")
            if not re.search(r"6\s*[–—-]\s*10\s*pm", visible):
                raise AssertionError("Generated HTML is missing the requested market time")
            if "<!doctype html" not in html.lower() or "</html>" not in html.lower():
                raise AssertionError("Generated HTML is incomplete")
            out.with_suffix(".html").write_text(html, encoding="utf-8")
        if args.case in ("all", "script"):
            result = request(args.endpoint,
                "Use find-skills to find the official skill-creator skill from anthropics/skills. "
                "Search, inspect it, install it using the inspected revision, and read its SKILL.md. "
                "I authorize that installation. Run its bundled scripts/aggregate_benchmark.py "
                "with args --help through skills_run. Do not write a replacement helper. "
                "Report the actual required positional argument and the optional output argument "
                "shown by the helper, with a short explanation of the files it produces.", args.max_tokens, cookie, args.temperature, args.seed)
            report["script"] = result
            verify(result, ("skills_search", "skills_inspect", "skills_install", "skills_read", "skills_run"), "skill-creator")
            if "benchmark_dir" not in result["answer"] or "--output" not in result["answer"]:
                raise AssertionError("Model answer does not report the actual helper interface")
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "report": str(out),
                          "error": report.get("error"),
                          "seconds": {key: report[key]["seconds"] for key in ("instructions", "script") if key in report}}, indent=2))


if __name__ == "__main__":
    main()
