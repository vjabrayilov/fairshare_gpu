PY=python

.PHONY: install
install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .

.PHONY: run
run:
	fairshare-gpu run --config configs/example_logical_vllm.yaml --out runs/logical_demo

.PHONY: analyze
analyze:
	fairshare-gpu analyze --run runs/logical_demo
	fairshare-gpu plot --run runs/logical_demo
