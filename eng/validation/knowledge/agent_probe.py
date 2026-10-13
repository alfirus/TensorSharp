#!/usr/bin/env python3
"""Real-model knowledge_open -> Python aggregation over a complete uploaded CSV.

Run against an already-started TensorAgentHost. This does not load/switch models,
change host settings, or treat a model's unsupported success claim as passing.
Reports, exact fixtures and raw SSE stay in the specified ignored output folder.
"""
import argparse
import csv
import hashlib
import io
import json
import pathlib
import random
import re
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=481731)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--retry", action="store_true", help="Allow one simpler fresh-session retry after recording failure.")
    args = parser.parse_args()
    if not 100 <= args.rows <= 5000:
        parser.error("--rows must be between 100 and 5000")
    connection = json.loads(pathlib.Path(args.connection).read_text())
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": connection.get("model"), "backend": connection.get("backend"),
        "executionBackend": connection.get("executionBackend"),
        "contextLength": connection.get("contextLength"), "attempts": [], "passed": False,
        "limitations": ["One synthetic integer CSV aggregation with an explicit tool-use prompt; not a general agent benchmark.",
            "Runs on the currently loaded model and configured code runtime; no model/device availability is counted as passing.",
            "Initial retrieval exposes only a bounded subset; success requires successful knowledge_open and shell events plus exact printed code output and final answer."],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def request(path, body=None, method=None, content_type="application/json"):
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode()
        req = urllib.request.Request(connection["baseUrl"] + path, data=body, method=method,
            headers={"Cookie": connection["cookie"], "Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.read()

    def js(path, body=None, method=None):
        return json.loads(request(path, body, method))

    def upload(collection, name, data):
        boundary = "knowledgeprobe" + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                f'Content-Type: text/csv\r\n\r\n').encode() + data + f"\r\n--{boundary}--\r\n".encode()
        return json.loads(request(f"/api/knowledge/collections/{collection}/documents", body,
            content_type=f"multipart/form-data; boundary={boundary}"))

    def events_from(raw):
        events = []
        for line in raw.decode().splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except ValueError:
                    pass
        return events

    rng = random.Random(args.seed)
    rows = [{"row_id": index + 1, "batch": f"batch-{index % 17:02d}", "amount_cents": rng.randint(-9371, 24583)}
            for index in range(args.rows)]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["row_id", "batch", "amount_cents"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    expected_sum = sum(row["amount_cents"] for row in rows)
    report["fixture"] = {"rows": args.rows, "seed": args.seed, "expectedTotalCents": expected_sum,
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (output / "measurement-ledger.csv").write_bytes(data)
    collection_id = None
    try:
        collection_id = js("/api/knowledge/collections", {"name": "Agent aggregate probe " + uuid.uuid4().hex[:8]})["collection"]["id"]
        report["collectionId"] = collection_id
        started = time.perf_counter()
        response = upload(collection_id, "measurement-ledger.csv", data)
        assert not response.get("errors"), response
        document = response["documents"][0]
        report["upload"] = {"seconds": time.perf_counter() - started, "document": document}
        assert document["status"] == "Ready", document
        hits = js("/api/knowledge/search", {"query": "amount cents", "collectionIds": [collection_id], "limit": 6})["results"]
        report["boundedRetrieval"] = {"resultCount": len(hits), "characters": sum(len(hit["text"]) for hit in hits),
            "chunkCount": document["chunkCount"], "sourceRows": args.rows}
        assert document["chunkCount"] > len(hits), "Fixture must exceed bounded retrieved evidence"
        original = request(f'/api/knowledge/collections/{collection_id}/documents/{document["id"]}/source?version={document["version"]}')
        assert original == data, "Persisted original bytes changed"
        save()
        for attempt in range(1, 3 if args.retry else 2):
            session = js("/api/sessions", {})["sessionId"]
            prompt = (
                "Compute an exact aggregate over the COMPLETE measurement-ledger.csv in the selected knowledge collection. "
                f"First call knowledge_open with collection_id='{collection_id}', document_id='{document['id']}', version={document['version']}. "
                "Use the returned private workspace path in a shell call running Python's csv.DictReader. "
                "Read EVERY row and sum integer amount_cents values; do not sum the retrieved snippets or calculate mentally. "
                "Have Python print ROWS=<row count> and TOTAL_CENTS=<integer sum>. "
                "Then report those two verified values and cite the original source."
            )
            if attempt == 2:
                prompt = (
                    f"Use knowledge_open now: collection_id={collection_id}, document_id={document['id']}, version={document['version']}. "
                    "Then run shell with python3: read the returned path with csv.DictReader, count all rows, "
                    "sum int(row['amount_cents']) for every row, and print ROWS=<count> and TOTAL_CENTS=<sum>. "
                    "Return the printed values. This task requires both tools; snippets are incomplete."
                )
            body = {"sessionId": session, "knowledgeCollections": [collection_id], "skills": [], "skills_discovery": False,
                "messages": [{"role": "user", "content": prompt}], "maxTokens": args.max_tokens, "think": False}
            (output / f"attempt-{attempt}-request.json").write_text(json.dumps(body, indent=2))
            started = time.perf_counter()
            raw = request("/api/chat", body)
            elapsed = time.perf_counter() - started
            (output / f"attempt-{attempt}.sse").write_bytes(raw)
            events = events_from(raw)
            answer = ""
            for event in events:
                if "replace" in event:
                    answer = event["replace"]
                elif "token" in event:
                    answer += event["token"]
            steps = [event for event in events if "skill_step" in event]
            output_text = "".join(str(event.get("text") or "") for event in events
                if event.get("tool") == "shell" and event.get("tool_progress") == "running")
            open_steps = [event for event in steps if event["skill_step"] == "knowledge_open" and event.get("ok") is True]
            code_steps = [event for event in steps if event["skill_step"] == "shell" and event.get("ok") is True]
            done = next((event for event in reversed(events) if event.get("done")), {})
            def contains_number(text, number):
                return re.search(r"(?<!\d)" + re.escape(str(number)) + r"(?!\d)", text.replace(",", "")) is not None
            code_exact = contains_number(output_text, args.rows) and contains_number(output_text, expected_sum)
            answer_exact = contains_number(answer, args.rows) and contains_number(answer, expected_sum)
            passed = bool(open_steps and code_steps and code_exact and answer_exact and not done.get("error") and not done.get("aborted"))
            entry = {"attempt": attempt, "sessionId": session, "seconds": elapsed, "passed": passed,
                "successfulKnowledgeOpen": bool(open_steps), "successfulShell": bool(code_steps),
                "codeOutputExact": code_exact, "answerExact": answer_exact, "answer": answer, "codeOutput": output_text,
                "steps": steps, "done": done, "authoritativeSources": [e["knowledgeSources"] for e in events if "knowledgeSources" in e]}
            report["attempts"].append(entry)
            report["passed"] = passed
            save()
            print(json.dumps({k: entry[k] for k in ("attempt", "seconds", "passed", "successfulKnowledgeOpen", "successfulShell", "codeOutputExact", "answerExact", "answer")}), flush=True)
            if passed:
                break
        if not report["passed"]:
            raise AssertionError("Agentic aggregation did not satisfy tool execution, exact code output and exact final-answer checks; see recorded attempts.")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["passed"] = False
        save()
        raise
    finally:
        # Delete only the collection created by this probe; source bytes and transcripts
        # remain in ignored evidence for review, without leaving user-visible test data.
        if collection_id is not None:
            try:
                report["cleanup"] = js(f"/api/knowledge/collections/{collection_id}", method="DELETE")
            except Exception as error:
                report["cleanupError"] = str(error)
            save()


if __name__ == "__main__":
    main()
