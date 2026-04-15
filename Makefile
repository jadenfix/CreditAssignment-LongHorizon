.PHONY: setup fmt lint type test smoke matrix-local matrix-tpu teacher-data clean docker-cpu docker-tpu

UV ?= uv
PY ?= $(UV) run python
OUT ?= artifacts/results/smoke
SEEDS ?= 2
BACKEND ?= local_nano
TASKS ?= math

setup:
	$(UV) sync --extra dev --extra ot

setup-all:
	$(UV) sync --extra dev --extra ot --extra hf --extra teacher --extra gcp

fmt:
	$(UV) run ruff format src tests scripts

lint:
	$(UV) run ruff check src tests scripts

type:
	$(UV) run pyright

test:
	$(UV) run pytest -q -m "not slow"

test-all:
	$(UV) run pytest -q

smoke:
	bash scripts/smoke_all_methods.sh

matrix-local:
	$(PY) scripts/run_paper_matrix.py --backend $(BACKEND) --tasks $(TASKS) --seeds $(SEEDS) --out $(OUT)
	$(PY) scripts/aggregate_results.py $(OUT) --base B3 --out $(OUT)/table.md

matrix-tpu:
	bash deploy/launch_tpu.sh

teacher-data:
	bash scripts/gen_all_teacher_data.sh

docker-cpu:
	docker build -f deploy/Dockerfile.cpu -t cts:cpu .

docker-tpu:
	docker build -f deploy/Dockerfile.tpu -t cts:tpu .

clean:
	rm -rf artifacts/results/smoke artifacts/runs .pytest_cache .ruff_cache
