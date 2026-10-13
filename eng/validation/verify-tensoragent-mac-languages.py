#!/usr/bin/env python3
"""Exercise native Mac Catalyst language switching with isolated settings.

Build TensorAgent/scripts/build-mac.sh, then run this tool. Each launch selects
a native Settings picker row through the existing Debug hook, returns to the
chat, and verifies the deferred reload and persisted choice on the next launch.
The real Mac Catalyst application runs; the user's settings and models are not
used. Evidence is restricted to ignored artifacts/ or docs/validation/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from urllib.request import Request, urlopen


REPO = Path(__file__).resolve().parents[2]
DEFAULT_APP = REPO / "TensorAgent/src/TensorAgent.Maui/bin/Debug/net10.0-maccatalyst/maccatalyst-arm64/TensorAgent.app"
LANGUAGES = ["zh-Hans", "zh-Hant", "ja", "ko", "es", "fr", "de", "en"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def request(connection: dict, path: str, body: dict | None = None) -> str:
    data = json.dumps(body).encode() if body is not None else None
    req = Request(connection["baseUrl"] + path, data=data,
                  headers={"Cookie": connection["cookie"], "Content-Type": "application/json"})
    with urlopen(req, timeout=10) as response:
        return response.read().decode()


def catalog(tag: str) -> dict:
    result = {}
    for path in (REPO / "TensorAgent/src/TensorAgent.Core/Localization" / tag).glob("*.json"):
        result.update(json.loads(path.read_text()))
    return result


def run_case(app: Path, output: Path, state: Path, initial: str, wanted: str,
             saved: str, timeout: float, pick: bool) -> dict:
    name = ("switch-" if pick else "relaunch-") + wanted
    log_path = output / (name + ".log")
    connection_path = state / "connection.json"
    connection_path.unlink(missing_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("TENSORAGENT_")}
    env["TENSORAGENT_VALIDATION_ROOT"] = str(state)
    if pick:
        env["TENSORAGENT_PICK_LANGUAGE"] = wanted
    row = {"name": name, "initial": initial, "expected": wanted, "status": "failed"}
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen([str(app / "Contents/MacOS/TensorAgent.Maui"),
                                        "-AppleLanguages", "(en-US)"], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)

            def wait_for(predicate, description: str):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    require(process.poll() is None,
                            f"App exited with {process.returncode} while {description}; see {log_path.name}")
                    text = log_path.read_text(errors="replace")
                    if predicate(text):
                        return text
                    time.sleep(.1)
                raise RuntimeError(f"Timed out {description}; see {log_path.name}")

            # A WebView-created session proves the chat has rendered far enough to
            # subscribe to native events. Dispatching from the Shell constructor is
            # too early on Mac Catalyst, where Shell.Current can still be null.
            text = wait_for(lambda text: connection_path.exists()
                            and "Created session via /api/sessions" in text, "waiting for the native chat")
            require(f"interface language {initial} (saved choice '{saved}'" in text,
                    "Startup did not restore the expected language choice")
            connection = json.loads(connection_path.read_text())
            require(connection["processId"] == process.pid, "Connection belongs to a previous app launch")
            row["startupRestoredChoice"] = saved
            started = time.perf_counter()
            request(connection, "/api/agent/events", {"type": "open-route", "route": "settings"})
            wait_for(lambda text: "open settings -> stack 2, top SettingsPage" in text
                     and (not pick or f"languagecheck pick {wanted} -> row" in text),
                     "opening settings and changing its native picker")
            row["settingsNavigationMs"] = (time.perf_counter() - started) * 1000
            settings = json.loads(request(connection, "/api/agent/settings"))
            require(settings["uiLanguage"] == wanted, "Settings did not save the picker choice")
            script = request(connection, "/i18n.js?lang=" + wanted)
            prefix = "window.TensorAgentI18n = "
            require(script.startswith(prefix), "Missing localization payload")
            payload = json.loads(script[len(prefix):script.index(";\n")])
            require(payload["lang"] == wanted, "Interface language differs from the selected language")
            require(payload["strings"] == catalog(wanted), "Served strings differ from this checkout")
            stored = json.loads((state / "Data/settings.json").read_text())
            require(stored["uiLanguage"] == wanted, "Language choice was not written to disk")
            row["catalogKeys"] = len(payload["strings"])
            (output / (name + "-i18n.json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            started = time.perf_counter()
            request(connection, "/api/agent/events", {"type": "open-route", "route": "main"})
            changed = initial != wanted
            text = wait_for(lambda text: "open main -> stack 1, top chat" in text
                            and (not changed or f"reloading the page in {wanted}" in text),
                            "returning to chat and reloading its language")
            row["chatNavigationAndReloadRequestMs"] = (time.perf_counter() - started) * 1000
            require(not any(error in text for error in (
                "open settings failed", "open main failed", "failed to repaint in the new language",
                "refresh on appearing failed", "page did not reload in the new language")),
                "Native navigation or language refresh logged a failure")
            page = request(connection, "/")
            require(f'/i18n.js?lang={wanted}"' in page, "Chat selects a stale localization bundle")
            # Keep pumping the actual event loop after the title/page refresh. The
            # reported native crash arrived on a later main-queue binding callback.
            latencies = []
            for _ in range(20):
                require(process.poll() is None, "App crashed after the language change")
                started = time.perf_counter()
                current = json.loads(request(connection, "/api/agent/settings"))
                latencies.append((time.perf_counter() - started) * 1000)
                require(current["uiLanguage"] == wanted, "Language changed unexpectedly")
                time.sleep(.1)
            row["settingsHttpLatencyMs"] = {"samples": len(latencies), "median": statistics.median(latencies),
                                            "p95": sorted(latencies)[18], "max": max(latencies)}
            row["nativeProcessSurvived"] = True
            row["deferredChatReloadObserved"] = changed
            row["status"] = "passed"
    except Exception as error:
        row["error"] = str(error)
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    app = args.app.resolve()
    require((app / "Contents/MacOS/TensorAgent.Maui").is_file(), "Build the Debug Mac Catalyst app first")
    require(args.timeout > 0, "Timeout must be positive")
    require(all(tag in LANGUAGES for tag in args.languages), "Unknown language")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.out or REPO / "artifacts/tensoragent-mac-languages" / stamp).resolve()
    require(any(output.is_relative_to(REPO / root) for root in ("artifacts", "docs/validation")),
            "Evidence must be inside ignored artifacts/ or docs/validation/")
    output.mkdir(parents=True, exist_ok=False)
    state = output / "isolated-app"
    assembly = app / "Contents/MonoBundle/TensorAgent.Maui.dll"
    report = {
        "app": str(app), "assemblySha256": hashlib.sha256(assembly.read_bytes()).hexdigest(),
        "startedUtc": stamp, "status": "failed", "cases": [],
        "limitations": [
            "One native Debug Mac Catalyst app on this Mac; no Release, Windows, iOS or physical iPhone run.",
            "The existing Debug hook selects the real native Settings picker row; it does not automate opening the OS picker popup.",
            "Verifies process survival, native navigation, reload requests, served translation tables and persistence. Visual layout is not assessed.",
            "Navigation timings include 100 ms polling granularity and stop at the native reload request, not the completed WebView paint.",
            "Settings HTTP timings are local endpoint latency; no inference model is loaded and no inference performance claim is made.",
        ],
    }
    initial, saved = "en", ""
    for wanted, pick in [(tag, True) for tag in args.languages] + [(args.languages[-1], False)]:
        row = run_case(app, output, state, initial, wanted, saved, args.timeout, pick)
        report["cases"].append(row)
        (output / "results.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
        if row["status"] != "passed":
            return 1
        initial, saved = wanted, wanted
    report["status"] = "passed"
    (output / "results.json").write_text(json.dumps(report, indent=2))
    print(f"Report: {output / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
