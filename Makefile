PYTHON ?= python3
PY := .venv/bin/python
HF := https://huggingface.co
LLAMA := data/llama-3.2-1b-instruct
GGUF := data/llama-3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct
MVU := LANES=4 WDEPTH=64 XDEPTH=8 YDEPTH=64 YPORTS=32
VPU := VK=1 WBITS=8 VDEPTH=1024 EDEPTH=256 XDEPTH=32
ATTN := VK=1 HD=64 NKV=1 REP=1 NLAYERS=1 MAXSEQ=16 ATT_TOK=4

.PHONY: setup models test check bench synth demo record clean

setup:
	$(PYTHON) -m venv .venv && $(PY) -m pip install -r requirements.txt

models:
	mkdir -p $(LLAMA) $(dir $(GGUF))
	for f in config.json tokenizer.json model.safetensors; do curl -sSL -o $(LLAMA)/$$f $(HF)/unsloth/Llama-3.2-1B-Instruct/resolve/main/$$f; done
	curl -sSL -o $(GGUF)-Q8_0.gguf $(HF)/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
	llama-quantize --allow-requantize $(GGUF)-Q8_0.gguf $(GGUF)-Q4_0.gguf Q4_0

test:
	$(PY) hw/run.py test --vk 1
	$(PY) hw/run.py test --vk 4
	$(PY) hw/run.py test --vk 4 --wbits 8
	$(PY) hw/run.py test --vk 4 --wbits 2

check:
	$(PY) hw/run.py check

bench:
	$(PY) hw/run.py bench llama-3.2-1b --lanes 256 --vk 4
	$(PY) hw/run.py bench llama-3.2-1b --lanes 1024 --vk 4
	$(PY) hw/run.py bench llama-3.2-1b --lanes 4096 --vk 16

synth:
	for lib in sky130hd asap7; do \
		$(PY) hw/synth.py mvu --lib $$lib --param $(MVU) && \
		$(PY) hw/synth.py vpu --lib $$lib --param $(VPU) && \
		$(PY) hw/synth.py attn --lib $$lib --param $(ATTN) || exit 1; \
	done

demo:
	$(PY) demo/server.py

record:
	$(PY) demo/server.py --record

clean:
	rm -rf hw/build hw/tests/results.xml hw/tests/__pycache__ hw/__pycache__ __pycache__
