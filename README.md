# HPML Project: FairShare-GPU: A Practical Demo of Multi-Tenant GPU Sharing for LLM Inference

## Team Information
- **Team Name**: FairShare
- **Members**:
    - Vahab Jabrayilov (vj2267)
    - Pulak Mehrorta (pm3371)

---

## 1. Problem Statement
LLM inference is increasingly deployed as a shared service where multiple tenants with heterogeneous prompt lengths, output lengths, and decoding parameters compete for the same GPU resources. The resulting contention can create head-of-line blocking, unpredictable tail latency, and "noisy neighbor" interference. This project investigates practical multi-tenant GPU sharing for LLM inference (focusing on NVIDIA L4) and provides a reproducible benchmark harness that quantifies throughput, latency percentiles (P95/P99), time-to-first-token (TTFT), and fairness under mixed workloads.

---

## 2. Model Description
We focus on single-GPU serving mechanisms (Logical, MPS, Simulated MIG) using open-source serving backends.

- **Models**: Llama-3 8B Instruct (primary), Mistral 7B Instruct (alternative).
- **Frameworks**: 
    - [vLLM](https://github.com/vllm-project/vllm) (for logical sharing)
    - [TGI / Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) (supported)
    - NVIDIA MPS (Multi-Process Service)
- **Hardware**: NVIDIA L4 GPU (24GB). Since L4 lacks MIG support, we implemented a simulated MIG baseline using resource limits.

---

## 3. Final Results Summary
Aggregate metrics across sharing modes (120s window):

| Metric | Logical Sharing | MPS Sharing | MIG-Simulated |
|:-----------------------|:---------------:|:-----------:|:-------------:|
| **Throughput** (tok/s) | 569.8 | 539.7 | 417.4 |
| **P95 Latency** (s) | 4.58 | 5.24 | 3.96 |
| **P99 Latency** (s) | 5.61 | 6.88 | 4.69 |
| **Jain Fairness** | 0.957 | 0.897 | 1.000 |
| **Device** | NVIDIA L4 (AWS g6.8xlarge) | NVIDIA L4 | NVIDIA L4 |

*Key Insight*: Logical sharing provides the highest throughput, while simulated MIG offers the best isolation (fairness and tail latency) at the cost of aggregate performance. MPS suffers from "noisy neighbor" effects with high tail latency.

---

## 4. Reproducibility Instructions

### A. Requirements
Install dependencies and the project package:
```bash
pip install -r requirements.txt
pip install -e .
```


### C. Training (Inference Benchmark)
This project is inference-only. To run the benchmark (e.g., Logical Sharing with vLLM):
```bash
# Example: Run logical sharing benchmark demo
fairshare-gpu run --config configs/example_logical_vllm.yaml --out runs/logical_demo
```
This command launches the server (vLLM), generates synthetic workload traffic, and logs metrics to the output directory.

---

### D. Evaluation
To analyze key metrics and generate plots from a run:
```bash
# Analyze results and generate plots (saved to runs/logical_demo/assets)
fairshare-gpu analyze --run runs/logical_demo
fairshare-gpu plot --run runs/logical_demo
```

---

### E. Quickstart: Minimum Reproducible Result
To reproduce our findings:

```bash
# Step 1: Set up environment
pip install -r requirements.txt
pip install -e .

# Step 2: Run a quick benchmark (Logical Sharing)
# This uses the example config included in the repo
fairshare-gpu run --config configs/example_logical_vllm.yaml --out runs/quickstart_logical

# Step 3: Analyze results
fairshare-gpu analyze --run runs/quickstart_logical
# Output will be in runs/quickstart_logical/summary.csv
```

---

## 5. Notes
- **Scripts**: All scripts are located in `scripts/`. Main logic is in `src/fairshare_gpu/`.
- **Configs**: See `configs/` for different sharing modes (Logical vs MPS).
- **Contact**: 
    - Vahab Jabrayilov (vj2267@columbia.edu)
    - Pulak Mehrorta (pm3371@columbia.edu)
