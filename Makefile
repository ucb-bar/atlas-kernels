# Run every kernel and library test suite, or generate one kernel program. NPU_MODEL names an
# npu_model checkout (the exactness checks use unmodified npu_model 47b8599 with numerics="rtl").
#   make test NPU_MODEL=...
#   make gen-projection_gemm NPU_MODEL=... ARGS='--shape 128x2048x2048 --traffic-reuse'
#   make coverage   (rebuild coverage/COVERAGE.md from the committed fixtures; needs PyYAML)
#   make bench-model MODEL=tinyllama OUT=... NPU_MODEL=... MODEL_PYTHON=... ATLAS_MLIR_BIN=...
# gen-<family> writes <stem>.S, the assembled <stem>.bin, and <stem>.json into build/<family>/.
PYTHON ?= python3
NPU_MODEL ?=
EXPORT_PYTHON ?= $(if $(TORCH_PYTHON),$(TORCH_PYTHON),$(PYTHON))
MODEL_PYTHON ?= $(TORCH_PYTHON)
ATLAS_MLIR_BIN ?=
# Adjacent pinned checkouts enable citation checks automatically. Explicit paths or
# CHECK_CITES=1 require both checkouts and fail if either is missing.
ATLAS_MLIR_ROOT ?= $(wildcard ../atlas-mlir)
MERLIN_ROOT ?= $(wildcard ../merlin)
CHECK_CITES ?= auto
SUITES := $(sort $(dir $(wildcard kernels/*/tests/test_*.py lib/tests/test_*.py)))
FAMILIES := $(patsubst kernels/%/tests/,%,$(filter kernels/%,$(SUITES)))
GENERATORS := $(patsubst kernels/%/gen.py,%,$(wildcard kernels/*/gen.py))
ARGS ?=
ENV := PYTHONDONTWRITEBYTECODE=1 $(if $(NPU_MODEL),PYTHONPATH=$(NPU_MODEL) ATLAS_NPU_MODEL_ROOT=$(NPU_MODEL))

WORKERS ?= 24
.PHONY: test test-lib test-coverage test-cites test-bench-model bench-model coverage $(addprefix test-,$(FAMILIES)) $(addprefix gen-,$(GENERATORS))
test: $(addprefix test-,$(FAMILIES)) test-lib test-coverage test-cites test-bench-model

test-lib:
	$(ENV) $(PYTHON) -m unittest discover -s lib/tests -t lib/tests -v

$(addprefix test-,$(FAMILIES)): test-%:
	$(ENV) $(PYTHON) -m unittest discover -s kernels/$*/tests -t kernels/$*/tests -v

$(addprefix gen-,$(GENERATORS)): gen-%:
	$(ENV) $(PYTHON) kernels/$*/gen.py --out build/$* $(ARGS)

coverage:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) coverage/build_table.py

test-coverage:
	$(ENV) $(PYTHON) -m unittest discover -s coverage/tests -t coverage/tests -v

test-cites:
	@if [ "$(CHECK_CITES)" = 1 ] || [ -n "$(ATLAS_MLIR_ROOT)$(MERLIN_ROOT)" ]; then \
	  test -n "$(ATLAS_MLIR_ROOT)" && test -n "$(MERLIN_ROOT)" || { echo 'citation checks require ATLAS_MLIR_ROOT and MERLIN_ROOT' >&2; exit 2; }; \
	  out=$$(mktemp -d); trap 'rm -rf "$$out"' EXIT; \
	  $(ENV) $(PYTHON) coverage/build_table.py --out "$$out" --check-cites "$(ATLAS_MLIR_ROOT)" "$(MERLIN_ROOT)" .; \
	else echo 'Citation checks unavailable: set ATLAS_MLIR_ROOT and MERLIN_ROOT, or require them with CHECK_CITES=1'; fi

# The plan tests also run under test-coverage; this target runs the npu_model execution tests with PYTHON and NPU_MODEL.
test-bench-model:
	$(ENV) $(PYTHON) -m unittest discover -s coverage/tests -t coverage/tests -p 'test_bench_model.py' -v

bench-model:
	@test -n "$(strip $(MODEL))" || { echo 'bench-model requires MODEL' >&2; exit 2; }
	@test -n "$(strip $(OUT))" || { echo 'bench-model requires OUT' >&2; exit 2; }
	@test -n "$(strip $(NPU_MODEL))" || { echo 'bench-model requires NPU_MODEL' >&2; exit 2; }
	@test -n "$(strip $(MODEL_PYTHON))" || { echo 'bench-model requires MODEL_PYTHON (Python >=3.14 with torch, NumPy, and npu_model)' >&2; exit 2; }
	@test -n "$(strip $(ATLAS_MLIR_BIN))" || { echo 'bench-model requires ATLAS_MLIR_BIN' >&2; exit 2; }
	$(ENV) "$(MODEL_PYTHON)" coverage/bench_model.py --coverage coverage/coverage.json --model "$(MODEL)" --model-root "$(NPU_MODEL)" --out "$(OUT)" --workers $(WORKERS) --atlas-mlir-bin "$(ATLAS_MLIR_BIN)" --min-tier stream-verified
