"""Validate measured token accounting, incremental delivery and failure gates."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("deepseek_maya_benchmark", Path(__file__).parents[1] / "deepseek-maya-http-benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def frame(text=None, finish=None, usage=None, reasoning=None):
    delta = {}
    if text is not None:
        delta["content"] = text
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    data = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        data["usage"] = usage
    return data


def wire(data):
    return ("data: " + ("[DONE]" if data == "[DONE]" else json.dumps(data, ensure_ascii=False)) + "\n\n").encode("utf-8")


class HttpServer:
    def __init__(self, respond):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    respond(self)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def headers(handler, status=200, content_type="text/event-stream"):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True


class StreamAccountingTests(unittest.TestCase):
    def test_usage_counts_not_frames_drive_decode_and_reasoning_counts_for_ttft(self):
        state = bench.StreamState()
        state.add({"choices": [{"delta": {"role": "assistant"}}]}, 2)
        state.add(frame(reasoning="Think "), 10)
        state.add(frame(text="Hello 世界"), 1010)
        state.add(frame(finish="stop", usage={"prompt_tokens": 64, "completion_tokens": 10,
                                            "prompt_tokens_details": {"cached_tokens": 32}}), 1020)
        state.add("[DONE]", 1021)
        state.validate()
        result = state.result()
        self.assertEqual(10, result["completion_tokens"])
        self.assertEqual(2, result["output_frames"])
        self.assertEqual(10, result["ttft_ms"])
        self.assertEqual(9, result["client_decode_tokens_per_second"])
        self.assertEqual(32, result["cached_prompt_tokens"])
        self.assertEqual("Hello 世界", result["output_text"])
        self.assertEqual("Think ", result["reasoning_text"])

    def test_one_batched_frame_has_no_observable_decode_window(self):
        state = bench.StreamState()
        state.add(frame(text="a whole batch of output", finish="length",
                        usage={"prompt_tokens": 5, "completion_tokens": 8}), 100)
        state.add("[DONE]", 110)
        state.validate()
        self.assertIsNone(state.result()["client_decode_tokens_per_second"])
        self.assertEqual(0, state.result()["decode_window_ms"])

    def test_missing_or_boolean_usage_never_falls_back_to_frame_counts(self):
        for usage in (None, {"prompt_tokens": True, "completion_tokens": 1},
                      {"prompt_tokens": 1, "completion_tokens": False},
                      {"prompt_tokens": 1, "completion_tokens": 1.0}):
            with self.subTest(usage=usage):
                state = bench.StreamState()
                state.add(frame(text="answer", finish="stop", usage=usage), 10)
                state.add("[DONE]", 11)
                with self.assertRaisesRegex(ValueError, "actual integer"):
                    state.validate()

    def test_content_after_finish_is_rejected_and_retained(self):
        state = bench.StreamState()
        state.add(frame(text="first", finish="stop"), 1)
        with self.assertRaisesRegex(ValueError, "after finish"):
            state.add(frame(text="second", finish="stop"), 2)
        self.assertEqual(2, len(state.events))

    def test_actual_tool_arguments_are_reassembled(self):
        state = bench.StreamState()
        for timestamp, args in ((1, '{"city":'), (2, '"Paris"}')):
            state.add({"choices": [{"delta": {"tool_calls": [{"index": 0,
                **({"id": "call1", "type": "function"} if timestamp == 1 else {}),
                "function": {**({"name": "weather"} if timestamp == 1 else {}), "arguments": args}}]}}]}, timestamp)
        state.add(frame(finish="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 4}), 3)
        state.add("[DONE]", 4)
        state.validate()
        tool = state.result()["tool_calls"][0]
        self.assertEqual("weather", tool["function"]["name"])
        self.assertEqual({"city": "Paris"}, json.loads(tool["function"]["arguments"]))

    def test_sse_multiline_crlf_comments_and_unterminated_events(self):
        payload = b': heartbeat\r\nevent: message\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: [DONE]\r\n\r\n'
        self.assertEqual([{"choices": []}, "[DONE]"], list(bench.sse_events(io.BytesIO(payload), time.monotonic() + 10)))
        with self.assertRaisesRegex(ValueError, "unterminated"):
            list(bench.sse_events(io.BytesIO(b'data: {}\n'), time.monotonic() + 10))

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_boolean_expected_counts(self):
        for text in ('{"ready":true,"ready":false}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                bench.strict_json(text)
        result = {"status": "complete", "finish_reason": "stop", "output_text": '{"count":true}'}
        self.assertEqual("failed", bench.check_correctness({"expected_json": {"count": 1}}, result)["status"])


class HttpDeliveryTests(unittest.TestCase):
    def test_short_flushed_frame_is_consumed_before_server_sends_final(self):
        consumed = threading.Event()
        observed = []
        original = bench.StreamState.add

        def add(state, event, elapsed):
            original(state, event, elapsed)
            if state.first_ms is not None:
                consumed.set()

        def respond(handler):
            self.assertEqual(True, handler.body["stream"])
            headers(handler)
            handler.wfile.write(wire(frame(text="579")))
            handler.wfile.flush()
            observed.append(consumed.wait(3))
            handler.wfile.write(wire(frame(finish="stop", usage={"prompt_tokens": 12, "completion_tokens": 2})))
            handler.wfile.write(wire("[DONE]"))
            handler.wfile.flush()

        with HttpServer(respond) as url, patch.object(bench.StreamState, "add", add):
            result = bench.run_request(url, {"stream": True}, 5)
        self.assertEqual([True], observed)
        self.assertEqual("complete", result["status"])
        self.assertEqual("579", result["output_text"])
        self.assertEqual(2, result["completion_tokens"])
        self.assertEqual(1, result["output_frames"])
        self.assertIsNone(result["client_decode_tokens_per_second"])
        self.assertEqual("passed", bench.check_correctness({"expected_text": "579"}, result)["status"])

    def test_truncated_or_malformed_stream_retains_output_and_raw_error(self):
        for tail, expected_error in ((b'', "without [DONE]"), (b'data: not json\n\n', "Expecting value"),
                                     (wire({"error": {"message": "out of memory"}}), "out of memory")):
            with self.subTest(tail=tail):
                def respond(handler):
                    headers(handler)
                    handler.wfile.write(wire(frame(text="partial")) + tail)
                    handler.wfile.flush()

                with HttpServer(respond) as url:
                    result = bench.run_request(url, {}, 5)
                self.assertEqual("failed", result["status"])
                self.assertIn(expected_error, result["error"])
                self.assertEqual("partial", result["output_text"])
                self.assertTrue(result["sse_lines"])

    def test_http_error_body_is_retained(self):
        def respond(handler):
            headers(handler, 503, "application/json")
            handler.wfile.write(b'{"error":"model not loaded"}')
            handler.wfile.flush()

        with HttpServer(respond) as url:
            result = bench.run_request(url, {}, 5)
        self.assertEqual(503, result["http_status"])
        self.assertEqual("failed", result["status"])
        self.assertIn("model not loaded", result["error_body"])

    def test_total_deadline_interrupts_a_trickling_close_delimited_stream(self):
        def respond(handler):
            headers(handler)
            for _ in range(40):
                handler.wfile.write(b': heartbeat\n\n')
                handler.wfile.flush()
                time.sleep(.025)

        with HttpServer(respond) as url:
            result = bench.run_request(url, {}, .15)
        self.assertEqual("failed", result["status"])
        self.assertEqual("TimeoutError", result["error_type"])
        self.assertLess(result["total_wall_ms"], 800)


class SuiteTests(unittest.TestCase):
    def test_expectations_need_clean_stop_and_preserve_unevaluated_quality(self):
        row = {"status": "complete", "finish_reason": "length", "output_text": "579"}
        self.assertEqual("failed", bench.check_correctness({"expected_text": "579"}, row)["status"])
        self.assertEqual("not_evaluated", bench.check_correctness({}, row)["status"])

    def test_workload_identity_excludes_alias_but_detects_engine_request_changes(self):
        case = {"messages": [{"role": "user", "content": "question"}]}
        first = bench.request_body(case, "alias1", 16, {"think": False})
        second = bench.request_body(case, "alias2", 16, {"think": False})
        self.assertEqual(bench.body_identity(first), bench.body_identity(second))
        second["think"] = True
        self.assertNotEqual(bench.body_identity(first), bench.body_identity(second))
        with self.assertRaisesRegex(ValueError, "cannot override"):
            bench.request_body(case, "alias1", 16, {"stream_options": {}})

    def test_output_must_be_ignored_and_existing_reports_are_not_overwritten(self):
        with TemporaryDirectory() as directory, patch.object(bench, "ROOT", Path(directory)):
            with self.assertRaisesRegex(ValueError, "ignored"):
                bench.output_path(Path(directory) / "tracked.json")
            path = bench.output_path(Path(directory) / "artifacts" / "result.json")
            path.write_text("existing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                bench.output_path(path)

    def test_summary_excludes_warmups_and_failed_requests_without_hiding_them(self):
        def row(phase, status, rate):
            return {"phase": phase, "status": status, "correctness": {"status": "not_evaluated"},
                    "ttft_ms": 20, "total_wall_ms": 120, "client_decode_tokens_per_second": rate,
                    "end_to_end_tokens_per_second": 30, "completion_tokens": 4, "prompt_tokens": 12,
                    "decode_window_ms": 100, "finish_reason": "length"}
        result = bench.summarize([row("warmup", "complete", 999), row("measured", "complete", 30),
                                  row("measured", "failed", 1000)])
        self.assertEqual(2, result["attempted"])
        self.assertEqual(1, result["failed"])
        self.assertEqual(30, result["client_decode_tokens_per_second"]["median"])
        self.assertEqual(30, result["pooled_client_decode_tokens_per_second"])

    def test_comparison_detects_tokenization_and_quality_failures(self):
        row = {"phase": "measured", "status": "complete", "correctness": {"status": "passed"},
               "ttft_ms": 20, "total_wall_ms": 120, "client_decode_tokens_per_second": 30,
               "end_to_end_tokens_per_second": 30, "completion_tokens": 4, "prompt_tokens": 12,
               "decode_window_ms": 100, "finish_reason": "stop"}
        baseline = {"cases_sha256": "a", "workload_identities": {"a": "b"}, "concurrency": 1,
                    "warmups": 1, "repeats": 1, "case_order": ["a"], "summary_by_case": {"a": bench.summarize([row])}}
        current = copy.deepcopy(baseline)
        self.assertTrue(bench.compare_reports(current, baseline)["cases"]["a"]["comparable_workload"])
        current["summary_by_case"]["a"]["prompt_token_counts"] = [13]
        current["summary_by_case"]["a"]["correctness"]["failed"] = 1
        comparison = bench.compare_reports(current, baseline)
        self.assertFalse(comparison["cases"]["a"]["comparable_workload"])
        self.assertFalse(comparison["performance_qualified"])

    def test_main_separates_warmups_retains_requests_and_rejects_smoke_failures(self):
        def respond(handler):
            headers(handler)
            handler.wfile.write(wire(frame(text="579", finish="stop", usage={"prompt_tokens": 12, "completion_tokens": 2})))
            handler.wfile.write(wire("[DONE]"))
            handler.wfile.flush()

        with TemporaryDirectory() as directory, patch.object(bench, "ROOT", Path(directory)), HttpServer(respond) as url:
            cases = Path(directory) / "cases.json"
            cases.write_text(json.dumps([{"name": "arithmetic", "messages": [{"role": "user", "content": "123+456?"}],
                                         "expected_text": "wrong"}]))
            output = Path(directory) / "artifacts" / "result.json"
            with patch("builtins.print"):
                code = bench.main(["--url", url, "--model", "alias", "--engine", "test", "--cases", str(cases),
                                   "--output", str(output), "--warmups", "1", "--repeats", "2", "--concurrency", "2"])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1, code)
            self.assertEqual(6, len(report["requests"]))
            self.assertEqual(4, report["summary_by_case"]["arithmetic"]["attempted"])
            self.assertEqual(4, report["summary_by_case"]["arithmetic"]["correctness"]["failed"])
            self.assertEqual("completed_with_failures", report["status"])
            self.assertFalse(report["quality_qualified"])


if __name__ == "__main__":
    unittest.main()
