# AIME / GPQA Tool-Use Benchmark

This repo runs math and QA benchmarks against an OpenAI-compatible chat endpoint, with a single Python execution tool exposed to the model.

It works with local vLLM deployments as long as the server exposes an OpenAI-compatible `/v1/chat/completions` API.

## Files

- `bench.py`
  Benchmark runner.
- `input/aime25/aime25.jsonl`
  AIME 2025 dataset.
- `input/gpqa_diamond/gpqa_diamond.jsonl`
  GPQA Diamond dataset.
- `test_openrouter_gpt_oss.sh`
  OpenRouter helper script. Not needed for local vLLM.

## Dataset Format

Each dataset row is JSONL with at least:

```json
{"id":"0","prompt":"...","answer":"123"}
```

Point `--input_dataset_dir` at a directory containing one or more `.jsonl` files.

## Requirements

Install the runtime dependencies in your Python environment:

```bash
pip install openai httpx rich sympy numpy
```

## Running Against Local vLLM

### 1. Start vLLM

Start vLLM with a model that supports tool calling over the OpenAI-compatible chat API.

Typical shape:

```bash
vllm serve /path/to/your/model \
  --host 0.0.0.0 \
  --port 8000
```

If your model needs special flags for tool calling or a custom chat template, add them at vLLM startup. The benchmark itself only assumes the endpoint is reachable at:

```text
http://HOST:PORT/v1
```

### 2. Run the benchmark

Minimal AIME run:

```bash
python bench.py \
  --api_base_url http://127.0.0.1:8000/v1 \
  --api_key dummy \
  --model_name your-model-name \
  --input_dataset_dir ./input/aime25 \
  --output_base_dir ./output \
  --thread_num 8 \
  --temperature 0 \
  --reasoning_effort medium \
  --run_id local_vllm_aime25_medium
```

Minimal GPQA run:

```bash
python bench.py \
  --api_base_url http://127.0.0.1:8000/v1 \
  --api_key dummy \
  --model_name your-model-name \
  --input_dataset_dir ./input/gpqa_diamond \
  --output_base_dir ./output \
  --thread_num 8 \
  --temperature 0 \
  --reasoning_effort medium \
  --run_id local_vllm_gpqa_medium
```

### 3. Compare `medium` vs `high`

```bash
python bench.py \
  --api_base_url http://127.0.0.1:8000/v1 \
  --api_key dummy \
  --model_name your-model-name \
  --input_dataset_dir ./input/aime25 \
  --output_base_dir ./output \
  --thread_num 8 \
  --temperature 0 \
  --reasoning_effort medium \
  --run_id local_vllm_aime25_medium

python bench.py \
  --api_base_url http://127.0.0.1:8000/v1 \
  --api_key dummy \
  --model_name your-model-name \
  --input_dataset_dir ./input/aime25 \
  --output_base_dir ./output \
  --thread_num 8 \
  --temperature 0 \
  --reasoning_effort high \
  --run_id local_vllm_aime25_high
```

The summaries will be written to:

- `output/summary_local_vllm_aime25_medium.json`
- `output/summary_local_vllm_aime25_high.json`

## Important Flags

- `--api_base_url`
  OpenAI-compatible base URL, usually `http://127.0.0.1:8000/v1` for local vLLM.
- `--api_key`
  Any non-empty string is usually fine for local vLLM.
- `--model_name`
  Model name sent in the chat request.
- `--thread_num`
  Number of benchmark worker threads.
- `--temperature`
  Sampling temperature.
- `--reasoning_effort`
  Sent through to the model request. Use this only if your deployed model/server understands it.
- `--code_exec_timeout_seconds`
  Timeout for each Python tool execution subprocess.
- `--worker_stall_timeout_seconds`
  Logs a warning if a worker appears stuck too long.
- `--run_id`
  Names the output directory. Reusing it allows resume behavior.
- `--allow_raw_tool_code_args`
  Optional recovery mode for malformed tool-call envelopes that send raw Python instead of `{"code": "..."}`.

## OpenRouter-Only Flags

Do not pass these for local vLLM:

- `--openrouter_provider_order`
- `--openrouter_provider_quantizations`
- `--openrouter_disable_provider_fallbacks`

They are only added to the request body when explicitly set.

## Output Layout

For a run like `--run_id local_vllm_aime25_medium`, outputs go under:

- `output/aime25/threads_res_local_vllm_aime25_medium`
  Per-thread JSONL outputs.
- `output/aime25/merge_jsonl_local_vllm_aime25_medium/aime25.jsonl`
  Merged result rows.
- `output/summary_local_vllm_aime25_medium.json`
  Final summary.

The summary includes:

- final accuracy
- answered / wrong / timeout / request-error counts
- function-call parse success rate
- function-call execution success rate
- repaired malformed tool-call count
- forced-finalization count

## Notes For Local vLLM

- The benchmark exposes one function tool named `python_exec`.
- Tool arguments are expected to be JSON with exactly one field:

```json
{"code":"print(2+2)"}
```

- If your model often emits malformed tool calls, you can try:

```bash
--allow_raw_tool_code_args
```

but keep it off if you want strict compliance testing.

- `reasoning_effort` is always sent by this harness. If your local deployment rejects unknown request fields, either:
  remove support server-side,
  patch the server to ignore it,
  or patch the harness if you want a model-specific request profile.

## Troubleshooting

### Empty or null responses

If you see repeated `content=None` with no tool calls, your server may not be returning standard assistant content for that model/template combination. Check:

- chat template
- tool-calling support in the served model
- whether the model actually supports OpenAI-style function calls in vLLM

### Frequent malformed tool calls

If the model emits raw code instead of `{"code": ...}`, either:

- keep strict mode and count that as a tool-calling failure, or
- rerun with `--allow_raw_tool_code_args` to tolerate raw valid Python strings

### Slow tail problems

Useful knobs:

```bash
--code_exec_timeout_seconds 90
--thread_num 4
--worker_stall_timeout_seconds 300
```

### Resume an interrupted run

Reuse the same run id:

```bash
python bench.py ... --run_id local_vllm_aime25_medium
```

Previously completed rows in that run directory will be skipped.
