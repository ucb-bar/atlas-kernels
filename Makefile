# Run every kernel and library test suite. NPU_MODEL names an npu_model checkout
# (the exactness checks use unmodified npu_model 47b8599 with numerics="rtl").
PYTHON ?= python3
NPU_MODEL ?=
SUITES := $(sort $(dir $(wildcard kernels/*/tests/test_*.py lib/tests/test_*.py)))
FAMILIES := $(patsubst kernels/%/tests/,%,$(filter kernels/%,$(SUITES)))
ENV := PYTHONDONTWRITEBYTECODE=1 $(if $(NPU_MODEL),PYTHONPATH=$(NPU_MODEL) ATLAS_NPU_MODEL_ROOT=$(NPU_MODEL))

.PHONY: test test-lib $(addprefix test-,$(FAMILIES))
test: $(addprefix test-,$(FAMILIES)) test-lib

test-lib:
	$(ENV) $(PYTHON) -m unittest discover -s lib/tests -t lib/tests -v

$(addprefix test-,$(FAMILIES)): test-%:
	$(ENV) $(PYTHON) -m unittest discover -s kernels/$*/tests -t kernels/$*/tests -v
