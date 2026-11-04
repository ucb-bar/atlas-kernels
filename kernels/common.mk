# Shared per-family interface. Invoke from kernels/<family> with make -C.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
FAMILY := $(notdir $(CURDIR))
PYTHON ?= python3
NPU_MODEL ?=
ATLAS_MLIR_BIN ?=
OUT ?= $(ROOT)/build/families
SEED ?= 7
RTL_SIMULATOR ?=
RTL_BUILD_RECEIPT ?=
RTL_MEMORY ?= both
RTL_TIMEOUT ?= 900
ENV ?= PYTHONDONTWRITEBYTECODE=1 $(if $(NPU_MODEL),PYTHONPATH=$(NPU_MODEL) ATLAS_NPU_MODEL_ROOT=$(NPU_MODEL))
RUN = $(ENV) "$(PYTHON)" "$(ROOT)/lib/family_interface.py"
CASE_ARGS = --family "$(FAMILY)" --out "$(OUT)" --seed "$(SEED)"

.DEFAULT_GOAL := gen
.PHONY: gen data test rtl
gen:
	$(RUN) gen $(CASE_ARGS)

# Phony dependencies deliberately regenerate on each invocation, including seed changes.
data: gen
	$(RUN) data $(CASE_ARGS)

test: data
	$(RUN) test $(CASE_ARGS) --model-root "$(NPU_MODEL)" --atlas-mlir-bin "$(ATLAS_MLIR_BIN)"

# Artifacts must be provisioned explicitly; this target never builds a simulator.
rtl: test
	$(RUN) rtl $(CASE_ARGS) --model-root "$(NPU_MODEL)" --atlas-mlir-bin "$(ATLAS_MLIR_BIN)" --simulator "$(RTL_SIMULATOR)" --build-receipt "$(RTL_BUILD_RECEIPT)" --memory "$(RTL_MEMORY)" --timeout "$(RTL_TIMEOUT)"
