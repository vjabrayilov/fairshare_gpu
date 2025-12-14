# Data

This folder contains a small prompt set (`prompts_small.jsonl`) for quick smoke tests.

For realistic experiments, we can replace it with a larger prompt corpus (e.g., ShareGPT-style),
as long as we keep the JSONL shape compatible:

- `{"prompt": "..."}`
- OR `{"text": "..."}`
- OR raw string lines

Then point `workload.kind: jsonl` and `workload.jsonl_path: data/<your_file>.jsonl` in the config.
