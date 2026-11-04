# Run every kernel and library test suite, or generate one kernel program. NPU_MODEL names an
# npu_model checkout (the exactness checks use unmodified npu_model 47b8599 with numerics="rtl").
#   make test NPU_MODEL=...
#   make gen-projection_gemm NPU_MODEL=... ARGS='--shape 128x2048x2048 --traffic-reuse'
#   make coverage   (rebuild coverage/COVERAGE.md from the committed fixtures; needs PyYAML)
# gen-<family> writes <stem>.S, the assembled <stem>.bin, and <stem>.json into build/<family>/.
PYTHON ?= python3
NPU_MODEL ?=
SUITES := $(sort $(dir $(wildcard kernels/*/tests/test_*.py lib/tests/test_*.py)))
FAMILIES := $(patsubst kernels/%/tests/,%,$(filter kernels/%,$(SUITES)))
GENERATORS := $(patsubst kernels/%/gen.py,%,$(wildcard kernels/*/gen.py))
ARGS ?=
ENV := PYTHONDONTWRITEBYTECODE=1 $(if $(NPU_MODEL),PYTHONPATH=$(NPU_MODEL) ATLAS_NPU_MODEL_ROOT=$(NPU_MODEL))

.PHONY: test test-lib test-coverage coverage $(addprefix test-,$(FAMILIES)) $(addprefix gen-,$(GENERATORS))
test: $(addprefix test-,$(FAMILIES)) test-lib test-coverage

test-lib:
	$(ENV) $(PYTHON) -m unittest discover -s lib/tests -t lib/tests -v

$(addprefix test-,$(FAMILIES)): test-%:
	$(ENV) $(PYTHON) -m unittest discover -s kernels/$*/tests -t kernels/$*/tests -v

$(addprefix gen-,$(GENERATORS)): gen-%:
	$(ENV) $(PYTHON) kernels/$*/gen.py --out build/$* $(ARGS)

coverage:
	PYTHONDONTWRITEBYTECODE=1 python3 coverage/build_table.py

test-coverage:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s coverage/tests -t coverage/tests -v
