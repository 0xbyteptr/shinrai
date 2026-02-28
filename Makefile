PY := $(shell command -v python3 || echo python)

.PHONY: help build merge tokenize train finetune chat clean

help:
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo "Targets:"
	@echo "  build         - scrape Wikipedia and build corpus"        \
	@echo "  merge         - merge multiple texts.pkl files"        \
	@echo "  tokenize      - tokenize a texts.pkl file"             \
	@echo "  train         - train a GPT model from scratch"         \
	@echo "  finetune      - fine-tune an existing model"           \
	@echo "  chat          - start interactive chat with a checkpoint" \
	@echo "  clean         - remove generated files"

build:
	$(PY) main.py build

merge:
	# set IN to space-separated list of inputs, OUT for output file
	$(PY) main.py merge $(IN) --out $(OUT)

tokenize:
	# set INPUT, TOKENS and TOKENIZER variables as needed
	# TOKENS may be .pkl or .bin
	$(PY) main.py tokenize --input $(INPUT) --out_tokens $(TOKENS) --out_tokenizer $(TOKENIZER)

train:
	# pass additional args via ARGS, e.g. make train ARGS="--epochs 10"
	$(PY) main.py train $(ARGS)

finetune:
	$(PY) main.py finetune $(ARGS)

chat:
	$(PY) main.py chat $(ARGS)

clean:
	rm -rf checkpoints *.pkl *.png
