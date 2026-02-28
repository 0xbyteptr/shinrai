# ShinrAI

GPT-from-scratch toy project. A minimal workflow for scraping text, tokenizing,
training a small transformer model, and chatting with it.

## Overview

- **build_corpus.py** – crawl Wikipedia, save texts.pkl
- **merge_data.py** – combine multiple corpora
- **tokenize_texts.py** – build tokenizer and tokenized data
- **train_gpt.py** – train a GPTSmall model from scratch
- **finetune_gpt.py** – continue training an existing checkpoint
- **legacy_chat.py** – simple interactive chatbot
- **main.py** – high‑level workflow manager with subcommands or menu
- **Makefile** – convenient shortcuts for the tasks

## Setup

```sh
python3 -m venv .venv        # create virtualenv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` installation may require CUDA/CPU-specific wheel. Visit
[pytorch.org](https://pytorch.org) for instructions.

## Typical workflow

1. **Scrape a corpus**
   ```sh
   make build
   # or: python main.py build --limit 50 --out my_texts.pkl
   ```

2. **Merge multiple corpora (optional)**
   ```sh
   make merge IN="texts1.pkl texts2.pkl" OUT=merged.pkl
   ```

3. **Tokenize** (output may be either `.pkl` or raw `.bin` tokens)
   ```sh
   make tokenize INPUT=texts.pkl
   # or
   make tokenize INPUT=texts.pkl TOKENS=tokenized_data.bin
   ```

4. **Train** (accepts `.pkl` or `.bin` data)
   ```sh
   make train ARGS="--epochs 10 --seq_len 128"
   ```

5. **Fine‑tune**
   ```sh
   make finetune ARGS="--model checkpoints/gpt_best.pt --data new_tokens.pkl"
   ```

6. **Chat**
   ```sh
   make chat ARGS="--model checkpoints/gpt_best.pt"
   ```

Alternatively, run `python main.py` without arguments to open an interactive menu.

## Development notes

- All scripts now expose a `main()` function and can be imported/programmatically
  invoked.
- `main.py` acts as a lightweight CLI dispatcher using `argparse`.
- `Makefile` provides shorthand targets and documents available options.
- The tokenizer is word-based and simple; recent updates improve parsing of
  contractions ("you'll", "don't") and prevent stray spaces around apostrophes.
  You can swap it for another implementation by editing `tokenizer_word.py`.

Feel free to extend, refactor or package this into a proper Python module.
