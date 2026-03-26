import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import bench


class _MockFunction:
    def __init__(self, arguments: str) -> None:
        self.name = "python_exec"
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, arguments: str) -> None:
        self.id = "tool-call-1"
        self.type = "function"
        self.function = _MockFunction(arguments)


class _MockMessage:
    def __init__(self, arguments: Optional[str] = None, content: Optional[str] = None) -> None:
        self.tool_calls = [_MockToolCall(arguments)] if arguments is not None else []
        self.content = content


class _MockChoice:
    def __init__(self, message: _MockMessage, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _MockResponse:
    def __init__(self, message: _MockMessage, finish_reason: str) -> None:
        self.choices = [_MockChoice(message, finish_reason)]


class _SequencedBenchmarkClient(bench.BenchmarkClient):
    def __init__(self, responses: List[_MockResponse]) -> None:
        self._responses = list(responses)
        self.verbose = False
        self.model_name = "unused"
        self.openrouter_provider_preferences = None

    def _call_model(
        self,
        messages: List[Dict[str, object]],
        use_tools: bool,
        temperature: float,
        reasoning_effort: str,
    ) -> _MockResponse:
        return self._responses.pop(0)

    def _exec_prefix(self) -> str:
        return ""

    def close(self) -> None:
        return None


class BenchAnswerExtractionTests(unittest.TestCase):
    def test_extract_final_answer_uses_last_boxed_value(self) -> None:
        text = r"\[\boxed{b=26}\] and \[\boxed{c=16\sqrt3}\] so \[\boxed{n=104}\]"
        self.assertEqual(bench.extract_final_answer(text), "104")

    def test_extract_final_answer_uses_trailing_scalar_inside_box(self) -> None:
        self.assertEqual(bench.extract_final_answer(r"\[\boxed{a+b=510}\]"), "510")


class BenchToolArgumentParsingTests(unittest.TestCase):
    def test_structured_tool_args_parse_in_strict_mode(self) -> None:
        message = _MockMessage(arguments='{"code": "print(1+1)"}')
        extracted = bench.try_extract_code_request(message, allow_raw_tool_code_args=False)
        self.assertEqual(extracted, "print(1+1)")

    def test_structured_tool_args_parse_in_tolerant_mode(self) -> None:
        message = _MockMessage(arguments='{"code": "print(1+1)"}')
        extracted = bench.try_extract_code_request(message, allow_raw_tool_code_args=True)
        self.assertEqual(extracted, "print(1+1)")

    def test_raw_tool_args_rejected_in_strict_mode(self) -> None:
        message = _MockMessage(arguments="print(1+1)")
        extracted = bench.try_extract_code_request(message, allow_raw_tool_code_args=False)
        self.assertIsNone(extracted)

    def test_raw_tool_args_accepted_in_tolerant_mode(self) -> None:
        message = _MockMessage(arguments="value = 21\nprint(value * 2)")
        extracted = bench.try_extract_code_request(message, allow_raw_tool_code_args=True)
        self.assertEqual(extracted, "value = 21\nprint(value * 2)")

    def test_invalid_raw_tool_args_rejected_in_tolerant_mode(self) -> None:
        message = _MockMessage(arguments="sp.nsolve(, x)")
        extracted = bench.try_extract_code_request(message, allow_raw_tool_code_args=True)
        self.assertIsNone(extracted)

    def test_structured_tool_args_repaired_when_extra_trailing_brace_present(self) -> None:
        message = _MockMessage(arguments='{"code": "print(1+1)"}\n}')
        diagnostics: Dict[str, object] = {}
        extracted = bench.try_extract_code_request(
            message,
            allow_raw_tool_code_args=False,
            diagnostics=diagnostics,
        )
        self.assertEqual(extracted, "print(1+1)")
        self.assertTrue(diagnostics.get("repaired_tool_call"))

    def test_non_repairable_structured_tool_args_still_fail(self) -> None:
        message = _MockMessage(arguments='{"code": "print(1+1)"')
        extracted = bench.try_extract_code_request(message, allow_raw_tool_code_args=False)
        self.assertIsNone(extracted)

    def test_legacy_content_parsing_unchanged(self) -> None:
        legacy_payload = '<|message|>{"code": "print(2+3)"}<|call|>'
        message = _MockMessage(content=legacy_payload)
        extracted = bench.try_extract_code_request(message, allow_raw_tool_code_args=False)
        self.assertEqual(extracted, "print(2+3)")


class BenchScoreSummaryTests(unittest.TestCase):
    def test_summarize_jsonl_scores_reports_function_call_success_rate(self) -> None:
        rows = [
            {
                "id": "ok",
                "pred_answer": "42",
                "gold_answer": "42",
                "is_scored": True,
                "is_correct": True,
                "end_type": "answered",
                "response": [
                    {"role": "assistant", "finish_reason": "tool_calls"},
                    {"role": "tool", "tool_status": "ok"},
                ],
            },
            {
                "id": "parse-fail",
                "pred_answer": None,
                "gold_answer": "0",
                "is_scored": True,
                "is_correct": False,
                "end_type": "no_final_answer",
                "response": [
                    {"role": "assistant", "finish_reason": "tool_calls"},
                ],
            },
            {
                "id": "tool-error",
                "pred_answer": None,
                "gold_answer": "0",
                "is_scored": True,
                "is_correct": False,
                "end_type": "no_final_answer",
                "response": [
                    {"role": "assistant", "finish_reason": "tool_calls"},
                    {"role": "tool", "tool_status": "error"},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "summary.jsonl"
            jsonl_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = bench.summarize_jsonl_scores(str(jsonl_path))

        counts = summary["counts"]
        self.assertEqual(counts["function_call_attempts"], 3)
        self.assertEqual(counts["function_call_parse_success"], 2)
        self.assertEqual(counts["function_call_parse_fail"], 1)
        self.assertEqual(counts["function_call_success"], 1)
        self.assertEqual(counts["function_call_error"], 1)
        self.assertEqual(summary["function_call_success_rate"], "1/3 = 33.33%")
        self.assertEqual(summary["function_call_parse_success_rate"], "2/3 = 66.67%")

    def test_summarize_jsonl_scores_reports_repaired_and_finalization_counters(self) -> None:
        rows = [
            {
                "id": "ok",
                "pred_answer": "42",
                "gold_answer": "42",
                "is_scored": True,
                "is_correct": True,
                "end_type": "answered",
                "repaired_tool_calls": 2,
                "forced_finalization_turns": 1,
                "response": [],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "summary.jsonl"
            jsonl_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = bench.summarize_jsonl_scores(str(jsonl_path))

        counts = summary["counts"]
        self.assertEqual(counts["repaired_tool_calls"], 2)
        self.assertEqual(counts["forced_finalization_turns"], 1)


class BenchProviderRoutingTests(unittest.TestCase):
    def test_build_openrouter_provider_preferences(self) -> None:
        preferences = bench.build_openrouter_provider_preferences(
            provider_order="deepinfra, clarifai",
            provider_quantizations="bf16, fp8",
            allow_fallbacks=False,
        )
        self.assertEqual(
            preferences,
            {
                "order": ["deepinfra", "clarifai"],
                "quantizations": ["bf16", "fp8"],
                "allow_fallbacks": False,
            },
        )

    def test_build_openrouter_provider_preferences_empty(self) -> None:
        preferences = bench.build_openrouter_provider_preferences()
        self.assertIsNone(preferences)


class BenchToolExecutionTests(unittest.TestCase):
    def test_execute_python_code_subprocess_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = bench.execute_python_code_subprocess(
                prefix="",
                prior_snippets=[],
                current_snippet="while True:\n    pass",
                scratch_dir=Path(tmp_dir),
                timeout_seconds=1,
            )

        self.assertEqual(result.status, "timeout")
        self.assertIn("timed out", result.error or "")

    def test_execute_python_code_subprocess_replays_state_without_duplicate_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scratch_dir = Path(tmp_dir)
            setup_snippet = "value = 21\nprint('setup')"
            setup_result = bench.execute_python_code_subprocess(
                prefix="",
                prior_snippets=[],
                current_snippet=setup_snippet,
                scratch_dir=scratch_dir,
                timeout_seconds=2,
            )
            replay_result = bench.execute_python_code_subprocess(
                prefix="",
                prior_snippets=[setup_snippet],
                current_snippet="print(value * 2)",
                scratch_dir=scratch_dir,
                timeout_seconds=2,
            )

        self.assertEqual(setup_result.status, "ok")
        self.assertEqual(setup_result.stdout, "setup\n")
        self.assertEqual(replay_result.status, "ok")
        self.assertEqual(replay_result.stdout, "42\n")

    def test_execute_python_code_subprocess_allows_large_integer_printing_with_default_prefix(self) -> None:
        client = bench.BenchmarkClient.__new__(bench.BenchmarkClient)
        prefix = bench.BenchmarkClient._exec_prefix(client)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = bench.execute_python_code_subprocess(
                prefix=prefix,
                prior_snippets=[],
                current_snippet="print(10**5000)",
                scratch_dir=Path(tmp_dir),
                timeout_seconds=2,
            )

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.stdout.startswith("1"))


class _FactoryIsolationClient:
    instance_count = 0

    def __init__(self) -> None:
        type(self).instance_count += 1

    def solve_one(self, user_input: str, **_: object) -> List[Dict[str, object]]:
        return [{"iter_num": 0}, {"end_type": "answered"}, {"final_answer": "1"}]

    def close(self) -> None:
        return None


class BenchWorkerIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        _FactoryIsolationClient.instance_count = 0

    def test_each_worker_builds_its_own_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task_queue: queue.Queue[str] = queue.Queue()
            worker_states: Dict[str, bench.WorkerState] = {}
            worker_state_lock = threading.Lock()
            workers = [
                bench.BenchmarkWorker(
                    thread_id=worker_id,
                    task_queue=task_queue,
                    target_filename_pattern=str(tmp_path / "worker*"),
                    client_factory=_FactoryIsolationClient,
                    max_tool_rounds=1,
                    max_total_turns=1,
                    temperature=0.0,
                    reasoning_effort="low",
                    scratch_root=tmp_path / "scratch",
                    code_exec_timeout_seconds=1,
                    allow_raw_tool_code_args=False,
                    worker_states=worker_states,
                    worker_state_lock=worker_state_lock,
                )
                for worker_id in range(2)
            ]

            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5)

        self.assertEqual(_FactoryIsolationClient.instance_count, 2)


class BenchControllerTests(unittest.TestCase):
    def test_solve_one_reserves_final_answer_turn_after_last_tool(self) -> None:
        client = _SequencedBenchmarkClient(
            [
                _MockResponse(_MockMessage(arguments='{"code": "print(248)"}'), "tool_calls"),
                _MockResponse(_MockMessage(content=r"\[\boxed{248}\]"), "stop"),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = client.solve_one(
                "question",
                max_tool_rounds=1,
                max_total_turns=1,
                temperature=0.0,
                reasoning_effort="low",
                sample_scratch_dir=Path(tmp_dir),
                code_exec_timeout_seconds=1,
            )

        final_answer = next(item["final_answer"] for item in result if "final_answer" in item)
        forced_turns = next(item["forced_finalization_turns"] for item in result if "forced_finalization_turns" in item)
        self.assertEqual(final_answer, "248")
        self.assertEqual(forced_turns, 1)

    def test_solve_one_blocks_extra_tool_call_and_still_finishes(self) -> None:
        client = _SequencedBenchmarkClient(
            [
                _MockResponse(_MockMessage(arguments='{"code": "print(111)"}'), "tool_calls"),
                _MockResponse(_MockMessage(arguments='{"code": "print(222)"}'), "tool_calls"),
                _MockResponse(_MockMessage(content=r"\[\boxed{111}\]"), "stop"),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = client.solve_one(
                "question",
                max_tool_rounds=1,
                max_total_turns=3,
                temperature=0.0,
                reasoning_effort="low",
                sample_scratch_dir=Path(tmp_dir),
                code_exec_timeout_seconds=1,
            )

        tool_messages = [item for item in result if item.get("role") == "tool"]
        final_answer = next(item["final_answer"] for item in result if "final_answer" in item)
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(final_answer, "111")


class _IntegrationClient:
    solve_calls = 0

    def __init__(self) -> None:
        pass

    def solve_one(
        self,
        user_input: str,
        sample_scratch_dir: Optional[Path] = None,
        code_exec_timeout_seconds: int = 1,
        **_: object,
    ) -> List[Dict[str, object]]:
        type(self).solve_calls += 1
        if "hang" in user_input:
            assert sample_scratch_dir is not None
            tool_result = bench.execute_python_code_subprocess(
                prefix="",
                prior_snippets=[],
                current_snippet="while True:\n    pass",
                scratch_dir=sample_scratch_dir,
                timeout_seconds=code_exec_timeout_seconds,
            )
            return [
                {
                    "role": "tool",
                    "code_snippet": "while True:\n    pass",
                    "tool_status": tool_result.status,
                    "tool_message": {"content": tool_result.error or tool_result.stdout},
                },
                {"iter_num": 1},
                {"end_type": "tool_timeout"},
            ]
        return [{"iter_num": 0}, {"end_type": "answered"}, {"final_answer": "42"}]

    def close(self) -> None:
        return None


class _RawToolArgsIntegrationClient:
    solve_calls = 0

    def __init__(self) -> None:
        pass

    def solve_one(
        self,
        user_input: str,
        allow_raw_tool_code_args: bool = False,
        **_: object,
    ) -> List[Dict[str, object]]:
        type(self).solve_calls += 1
        raw_message = _MockMessage(arguments="value = 6\nprint(value * 7)")
        extracted = bench.try_extract_code_request(
            raw_message,
            tag="integration",
            allow_raw_tool_code_args=allow_raw_tool_code_args,
        )
        if extracted is None:
            return [{"iter_num": 0}, {"end_type": "no_final_answer"}]
        return [{"iter_num": 0}, {"end_type": "answered"}, {"final_answer": "42"}]

    def close(self) -> None:
        return None


class _ResumeClient:
    solve_calls = 0

    def __init__(self) -> None:
        pass

    def solve_one(self, user_input: str, **_: object) -> List[Dict[str, object]]:
        type(self).solve_calls += 1
        return [{"iter_num": 0}, {"end_type": "answered"}, {"final_answer": "7"}]

    def close(self) -> None:
        return None


class BenchRunBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        _IntegrationClient.solve_calls = 0
        _RawToolArgsIntegrationClient.solve_calls = 0
        _ResumeClient.solve_calls = 0

    def test_run_benchmark_completes_when_one_sample_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            output_dir.mkdir()
            dataset_path = dataset_dir / "samples.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "hang", "prompt": "hang sample", "answer": "0"}),
                        json.dumps({"id": "ok", "prompt": "normal sample", "answer": "42"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            bench.run_benchmark(
                input_dataset_dir=str(dataset_dir),
                output_base_dir=str(output_dir),
                model_name="unused",
                api_base_url="unused",
                api_key="unused",
                thread_num=2,
                max_tool_rounds=1,
                max_total_turns=1,
                temperature=0.0,
                reasoning_effort="low",
                http_timeout_seconds=1,
                code_exec_timeout_seconds=1,
                worker_stall_timeout_seconds=2,
                run_id="integration",
                client_factory=_IntegrationClient,
            )

            merged_path = output_dir / "samples" / "merge_jsonl_integration" / "samples.jsonl"
            rows = [json.loads(line) for line in merged_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(_IntegrationClient.solve_calls, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["end_type"] for row in rows}, {"answered", "tool_timeout"})

    def test_run_benchmark_raw_tool_args_need_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            output_dir.mkdir()
            dataset_path = dataset_dir / "samples.jsonl"
            dataset_path.write_text(
                json.dumps({"id": "raw", "prompt": "raw args sample", "answer": "42"}) + "\n",
                encoding="utf-8",
            )

            bench.run_benchmark(
                input_dataset_dir=str(dataset_dir),
                output_base_dir=str(output_dir),
                model_name="unused",
                api_base_url="unused",
                api_key="unused",
                thread_num=1,
                max_tool_rounds=1,
                max_total_turns=1,
                temperature=0.0,
                reasoning_effort="low",
                http_timeout_seconds=1,
                code_exec_timeout_seconds=1,
                worker_stall_timeout_seconds=2,
                allow_raw_tool_code_args=False,
                run_id="raw-strict",
                client_factory=_RawToolArgsIntegrationClient,
            )
            strict_rows = [
                json.loads(line)
                for line in (
                    output_dir / "samples" / "merge_jsonl_raw-strict" / "samples.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            bench.run_benchmark(
                input_dataset_dir=str(dataset_dir),
                output_base_dir=str(output_dir),
                model_name="unused",
                api_base_url="unused",
                api_key="unused",
                thread_num=1,
                max_tool_rounds=1,
                max_total_turns=1,
                temperature=0.0,
                reasoning_effort="low",
                http_timeout_seconds=1,
                code_exec_timeout_seconds=1,
                worker_stall_timeout_seconds=2,
                allow_raw_tool_code_args=True,
                run_id="raw-tolerant",
                client_factory=_RawToolArgsIntegrationClient,
            )
            tolerant_rows = [
                json.loads(line)
                for line in (
                    output_dir / "samples" / "merge_jsonl_raw-tolerant" / "samples.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(strict_rows[0]["end_type"], "no_final_answer")
        self.assertEqual(tolerant_rows[0]["end_type"], "answered")
        self.assertEqual(tolerant_rows[0]["pred_answer"], "42")

    def test_run_benchmark_resumes_with_same_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            dataset_dir.mkdir()
            output_dir.mkdir()
            dataset_path = dataset_dir / "samples.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "1", "prompt": "sample one", "answer": "7"}),
                        json.dumps({"id": "2", "prompt": "sample two", "answer": "7"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            bench.run_benchmark(
                input_dataset_dir=str(dataset_dir),
                output_base_dir=str(output_dir),
                model_name="unused",
                api_base_url="unused",
                api_key="unused",
                thread_num=2,
                max_tool_rounds=1,
                max_total_turns=1,
                temperature=0.0,
                reasoning_effort="low",
                http_timeout_seconds=1,
                code_exec_timeout_seconds=1,
                worker_stall_timeout_seconds=2,
                run_id="resume",
                client_factory=_ResumeClient,
            )
            first_call_count = _ResumeClient.solve_calls

            bench.run_benchmark(
                input_dataset_dir=str(dataset_dir),
                output_base_dir=str(output_dir),
                model_name="unused",
                api_base_url="unused",
                api_key="unused",
                thread_num=2,
                max_tool_rounds=1,
                max_total_turns=1,
                temperature=0.0,
                reasoning_effort="low",
                http_timeout_seconds=1,
                code_exec_timeout_seconds=1,
                worker_stall_timeout_seconds=2,
                run_id="resume",
                client_factory=_ResumeClient,
            )

        self.assertEqual(first_call_count, 2)
        self.assertEqual(_ResumeClient.solve_calls, 2)


if __name__ == "__main__":
    unittest.main()
