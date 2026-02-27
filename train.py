#!/usr/bin/env python
"""train.py — Train a shinrai character-level RNN model.

Usage examples:

    # Single Wikipedia article
    python train.py --url https://en.wikipedia.org/wiki/Philosophy

    # Curated seed corpus, save checkpoint, resume automatically
    python train.py --use_seed_articles --epochs 30 --save_checkpoint model.pt

    # Local file, GRU backend, bigger model
    python train.py --text_file corpus.txt --cell_type gru \\
                    --hidden_size 1024 --num_layers 3 --epochs 50

    # Resume with more epochs
    python train.py --use_seed_articles --save_checkpoint model.pt --epochs 60
"""

import argparse
import sys

import torch

from shinrai.config import (
    add_checkpoint_args,
    add_data_args,
    add_generate_args,
    add_model_args,
    add_train_args,
    config_from_namespace,
)
from shinrai.generate import generate
from shinrai.logging import log
from shinrai.trainer import run_training, _resolve_device




def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train",
        description="shinrai — train a character-level RNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_args(p)
    add_model_args(p)
    add_train_args(p)
    add_checkpoint_args(p)
    add_generate_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg  = config_from_namespace(args)

    log.banner("shinrai — character-level RNN trainer")

    # delegate to shared helper which encapsulates the traditional
    # "train.py" pipeline.  The helper also handles logging banner and
    # checkpoint defaults so the CLI remains lightweight.

    log.rule("Training")
    from shinrai.trainer import run_training

    model, vocab = run_training(cfg)
    log.rule("Done")

    # ── 5. Interactive generation ─────────────────────────────────────────────
    log.info("\nEnter seed text to generate (or 'exit' to quit).\n")
    while True:
        try:
            prompt = input("Seed> ").strip()
        except (EOFError, KeyboardInterrupt):
            log.info("\nBye!")
            break
        if not prompt:
            continue
        if prompt.lower() == "exit":
            break
        device = _resolve_device(cfg.train.device)
        output = generate(
            model, vocab, prompt,
            length=cfg.generate.gen_length,
            temperature=cfg.generate.temperature,
            top_p=cfg.generate.top_p,
            seq_length=cfg.model.seq_length,
            device=device,
        )
        log.success("\n" + output + "\n")


if __name__ == "__main__":
    main()
