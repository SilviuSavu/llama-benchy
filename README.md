# LLM Benchmark Script

This script benchmarks OpenAI-compatible LLM endpoints, generating statistics similar to `llama-bench`.

## Features

- Measures Prompt Processing (pp) and Token Generation (tg) speeds.
- Reports Time To First Token (TTFT), Time To First Response (TTFR), and End-to-End TTFT.
- Supports configurable prompt length (`--pp`), generation length (`--tg`), and context depth (`--depth`).
- Can run multiple iterations (`--runs`) and report mean ± std.
- Uses HuggingFace tokenizers for accurate token counts.
- Can download a book from Project Gutenberg to use as source text for prompts, or use synthetic data.
- Supports executing a command after each run (e.g., to clear cache).
- Configurable latency measurement mode.

## Installation

1.  Create a virtual environment:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

## Usage

```bash
python llm_bench.py --base-url <ENDPOINT_URL> --model <MODEL_NAME> --pp <PROMPT_TOKENS> --tg <GEN_TOKENS> [OPTIONS]
```

### Arguments

-   `--base-url`: OpenAI compatible endpoint URL (Required).
-   `--api-key`: API Key (Default: "EMPTY").
-   `--model`: Model name (Required).
-   `--tokenizer`: HuggingFace tokenizer name (Defaults to model name).
-   `--pp`: List of prompt processing token counts (Required).
-   `--tg`: List of token generation counts (Required).
-   `--depth`: List of context depths (Default: [0]).
-   `--runs`: Number of runs per test (Default: 3).
-   `--no-cache`: Ensure unique requests to avoid prefix caching.
-   `--post-run-cmd`: Command to execute after each test run.
-   `--book-url`: URL of a book to use for text generation.
-   `--latency-mode`: Method to measure latency: 'models' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement).
-   `--no-warmup`: Skip warmup phase.

### Metrics

The script outputs a table with the following metrics. All time measurements are in milliseconds (ms).

#### Latency Adjustment
The script attempts to estimate network latency to provide "server-side" processing times.
- **Latency**: Measured based on `--latency-mode`.
  - `models`: Time to fetch `/models` (from sending request to getting first byte of the response).
  - `generation`: Time to generate 1 token (from sending request to getting first byte of the response).
  - `none`: Assumed to be 0.
- This measured latency is subtracted from `e2e_ttft` and `ttfr` to calculate `ttft` and `ttfr`.

#### Table Columns

-   **`t/s` (Tokens per Second)**:
    -   **For Prompt Processing (pp)**: Calculated as `Total Prompt Tokens / TTFR`. This represents the prefill speed.
    -   **For Token Generation (tg)**: Calculated as `(Total Generated Tokens - 1) / (Time of Last Token - Time of First Token)`. This represents the decode speed, excluding the first token latency.

-   **`ttft (ms)` (Time To First Token)**:
    -   Calculation: `End-to-End TTFT - Estimated Latency`.
    -   Represents the time the server takes to process the prompt and generate the *first content token*.

-   **`ttfr (ms)` (Time To First Response)**:
    -   Calculation: `(Time of First Response Chunk - Start Time) - Estimated Latency`.
    -   Represents the time until the client receives *any* response from the server (including empty chunks or role definitions), adjusted for latency. Used for calculating Prompt Processing speed.

-   **`e2e_ttft (ms)` (End-to-End Time To First Token)**:
    -   Calculation: `Time of First Content Token - Start Time`.
    -   The total time perceived by the client from sending the request to seeing the first generated content.

### Example

```bash
python llm_bench.py \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --pp 128 256 \
  --tg 32 64 \
  --depth 0 1024
```

This will run benchmarks for all combinations of pp (128, 256), tg (32, 64), and depth (0, 1024).
