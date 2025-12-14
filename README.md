# FairShare-GPU

**FairShare-GPU** is a practical benchmarking + demo tool for **multi-tenant GPU sharing during LLM inference**.
It is designed to help **measure isolation, utilization, tail-latency, and fairness** when multiple tenants
share a single NVIDIA GPU using:

1. **Spatial partitioning** via **MIG** (multiple serving instances, each bound to a MIG slice)
2. **Process/time sharing** via **MPS / time-slicing** (multiple serving instances/processes share one GPU)
3. **Logical sharing** inside a serving framework (e.g., a single vLLM server handling all tenants)

The repository does **not** implement model serving itself. Instead, it provides:
- a **multi-tenant load generator** (async, streaming) for vLLM/TGI endpoints
- a **GPU telemetry sampler** (NVML) for utilization/memory/power/temperature
- a **results format** (JSONL + CSV) and **analysis/plotting scripts**

---

## Repository layout

```
fairshare-gpu/
  configs/                  # Example experiment configs (YAML)
  data/                     # Small prompt sets + helpers
  report/                   # LaTeX report template (IEEE-style) + assets folder
  scripts/                  # Convenience launchers
  src/fairshare_gpu/        # Python package
  pyproject.toml            # Python deps + entrypoints
```

---

## Prerequisites

### Hardware / system
- **NVIDIA GPU** with a recent driver and CUDA runtime.
- For **MIG mode**: an **A100/H100**-class GPU that supports MIG.
- For **MPS mode**: any CUDA GPU that supports MPS.

### Serving backends
This repo assumes you run a serving backend separately, then benchmark it:

- **vLLM**: run the OpenAI-compatible server.
- **Hugging Face TGI**: use the native `/generate` endpoint.


---

## Install (client + analysis tooling)

Create a Python 3.10+ environment and install this repo in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -e .
```

This installs the benchmarking/analysis dependencies (httpx, pandas, matplotlib, pynvml, etc.).
Need to install **vLLM/TGI separately** according to their docs.

---

## Quickstart (single server, multi-tenant logical sharing)

### 1) Start vLLM OpenAI server (example)

In another terminal:

```bash
# Example: vLLM OpenAI-compatible server (adjust model + dtype for your GPU)
python -m vllm.entrypoints.openai.api_server   --host 0.0.0.0 --port 8000   --model meta-llama/Meta-Llama-3-8B-Instruct   --dtype bfloat16   --max-num-batched-tokens 8192
```

Verify it responds:

```bash
curl http://localhost:8000/v1/models
```

### 2) Run a benchmark

```bash
fairshare-gpu run --config configs/example_logical_vllm.yaml --out runs/logical_demo
```

Outputs:
- `runs/logical_demo/requests.jsonl`  (per-request measurements)
- `runs/logical_demo/gpu.csv`         (NVML telemetry, if enabled)
- `runs/logical_demo/config.resolved.yaml`

### 3) Analyze + plot

```bash
fairshare-gpu analyze --run runs/logical_demo
fairshare-gpu plot --run runs/logical_demo
```

This creates:
- `runs/logical_demo/analysis/summary.json`
- `runs/logical_demo/figures/*.png`

---

## Running MIG experiments

**Goal:** start *one server per tenant* and bind each server to a **different MIG device**.

High-level steps:
1. Enable MIG and create GPU/compute instances (`nvidia-smi mig ...`)
2. Run `nvidia-smi -L` to list MIG device UUIDs
3. Launch one vLLM/TGI server per tenant with `CUDA_VISIBLE_DEVICES=<MIG_UUID>`
4. Run with a config that points each tenant to its server endpoint

Example server launch (per tenant):

```bash
export CUDA_VISIBLE_DEVICES="MIG-xxxxxxxx-xxxx-...."   # one slice UUID
python -m vllm.entrypoints.openai.api_server --port 8001 --model ... &
```

Then run:

```bash
fairshare-gpu run --config configs/example_mig_vllm.yaml --out runs/mig_demo
```

> Notes
> - MIG changes require GPU reset and typically admin privileges.
> - MIG profiles and commands vary by GPU model; use `nvidia-smi mig -lgip` to list supported profiles.

---

## Running MPS experiments

**Goal:** run multiple processes/servers on the *same GPU* while the **MPS daemon** enables concurrent kernel execution.

Typical flow:
1. Start the MPS control daemon
2. Optionally set per-process `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` to approximate fairness
3. Launch one server per tenant (different ports)
4. Benchmark

Example:

```bash
# Start MPS daemon (paths can be anywhere writable)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
nvidia-cuda-mps-control -d

# Tenant A server (example)
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
python -m vllm.entrypoints.openai.api_server --port 8001 --model ... &

# Tenant B server (example)
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
python -m vllm.entrypoints.openai.api_server --port 8002 --model ... &
```

Then run:

```bash
fairshare-gpu run --config configs/example_mps_vllm.yaml --out runs/mps_demo
```

Stop MPS:

```bash
echo quit | nvidia-cuda-mps-control
```

---

## Configuration files (YAML)

Benchmarks are driven by a single YAML config.

Key fields:
- `backend`: `vllm_openai` or `tgi_generate`
- `model`: model name for token counting (HF tokenizer)
- `tenants[]`: each tenant has an `id`, an `endpoint`, and a workload definition
- `workload`: dataset source + prompt mix (synthetic or JSONL)
- `benchmark`: duration, warmup, streaming, rate/concurrency controls
- `telemetry`: NVML sampling settings

See `configs/` for examples.

---

## Metrics computed

From `requests.jsonl`, we compute (per tenant and aggregated):
- **Throughput:** output tokens/s and requests/s
- **Latency:** P50 / P95 / P99 end-to-end latency
- **TTFT:** time-to-first-token (streaming mode)
- **SLO attainment:** fraction of requests meeting a latency SLO
- **Fairness:** Jain’s fairness index over per-tenant throughput
- **Interference:** slowdown vs a baseline run 

GPU telemetry (from `gpu.csv`):
- GPU utilization %, memory utilization %, memory used/total, power, temperature


---

## Common issues / troubleshooting

- **Streaming hangs / timeouts:** increase `benchmark.timeout_s` and ensure server supports streaming.
- **Tokenizer download is slow:** set `HF_HOME` to a fast disk cache, or pre-download models.
- **NVML errors:** install a recent NVIDIA driver; `pynvml` reads telemetry via the driver.
- **MIG not available:** use `logical` + `mps` modes and still quantify tail-latency & fairness.

