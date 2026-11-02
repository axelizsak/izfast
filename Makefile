PYTHON ?= python3
PY := .venv/bin/python
HF := https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main
QWEN := data/qwen3.5-4b
LANES := WDEPTH=64 HDEPTH=64 XDEPTH=8 YDEPTH=64 YPORTS=32 SDEPTH=16
VPU1 := VK=1 WBITS=8 VDEPTH=1024 EDEPTH=256 XDEPTH=32 GAINS=1 GDEPTH=16 EPS=4295
DNU1 := VK=1 HK=2 HV=4 DK=32 DV=32 CK=4 NLIN=1 SH=1 LS=5793 EPS=4295

.PHONY: setup models test check bench synth demo record clean

setup:
	$(PYTHON) -m venv .venv && $(PY) -m pip install -r requirements.txt

models:
	mkdir -p $(QWEN)
	for f in config.json tokenizer.json model.safetensors.index.json model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors; do \
		curl -sSL --fail -o $(QWEN)/$$f $(HF)/$$f || exit 1; done
	$(PY) hw/run.py emu qwen3.5-4b --wbits 4 --hbits 8
	$(PY) hw/eval.py
	rm $(QWEN)/model.safetensors-*

test:
	$(PY) hw/run.py test --vk 1 --wbits 4 --hbits 8 --model tiny-qwen
	$(PY) hw/run.py test --vk 1 --wbits 4 --hbits 8 --model tiny-qwen --sh 2
	$(PY) hw/run.py test --vk 4 --wbits 4 --hbits 8 --model tiny-qwen-w
	$(PY) hw/run.py test --vk 4 --wbits 4 --hbits 8 --model tiny-qwen-w --sh 4
	$(PY) hw/run.py test --vk 1
	$(PY) hw/run.py test --vk 4
	$(PY) hw/run.py test --vk 4 --wbits 8
	$(PY) hw/run.py test --vk 4 --wbits 2 --hbits 8 --model tiny-bitnet
	$(PY) hw/run.py test --vk 1 --lora 32

check:
	$(PY) hw/run.py check --model qwen3.5-4b --wbits 4 --hbits 8 --tokens 4

bench:
	$(PY) hw/run.py bench qwen3.5-4b --lanes 1024 --vk 4 --att-tok 4 --sh 8 --ctx 0,511,1023

synth:
	for l in 4 8; do $(PY) hw/synth.py mvu --lib asap7 --tag w4h8_l$$l --param LANES=$$l WBITS=4 HBITS=8 $(LANES) || exit 1; done
	$(PY) hw/synth.py dnu --lib asap7 --tag vk1 --param $(DNU1)
	$(PY) hw/synth.py vpu --lib asap7 --tag vk1 --param $(VPU1)

demo:
	$(PY) demo/server.py

record:
	$(PY) demo/server.py --record

clean:
	rm -rf hw/build hw/tests/results.xml hw/tests/__pycache__ hw/__pycache__ __pycache__
