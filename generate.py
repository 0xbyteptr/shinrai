#!/usr/bin/env python
"""generate.py — Sample text from a saved shinrai checkpoint.

Usage:

    python generate.py --checkpoint model.pt --seed "the meaning of"
    python generate.py --checkpoint model.pt --seed "once upon" --length 600 --temperature 1.0
    python generate.py --checkpoint model.pt  # interactive mode
"""

import argparse
import sys

import torch

from shinrai.data import Vocabulary
from shinrai.generate import generate as _generate
from shinrai.logging import log
from shinrai.model import CharRNN
from shinrai.trainer import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate",
        description="shinrai — generate text from a trained checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="Path to a .pt checkpoint produced by train.py")
    p.add_argument("--seed", default=None,
                   help="Seed text (omit for interactive mode)")
    p.add_argument("--length",      type=int,   default=400,
                   help="Number of characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p",       type=float, default=0.9,
                   help="Nucleus sampling p (1.0 = disabled)")
    p.add_argument("--device",      default=None)
    return p


def load_model_from_checkpoint(
    path: str, device: torch.device
) -> tuple[CharRNN, Vocabulary]:
    """Reconstruct model and vocabulary entirely from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)

    model_cfg = ckpt.get("model_config")
    if model_cfg is None:
        log.error(
            "Checkpoint does not contain 'model_config'. "
            "Re-train with the current version of shinrai."
        )
        sys.exit(1)

    chars = ckpt.get("chars")
    if not chars:
        log.error("Checkpoint does not contain vocabulary ('chars').")
        sys.exit(1)

    vocab = Vocabulary.from_chars(chars)
    # Patch vocab_size in case it was stored inconsistently
    model_cfg["vocab_size"] = len(vocab)

    model = CharRNN.from_config(model_cfg, dropout=0.0).to(device)
    state = ckpt.get("model_state", {})
    # some training-time wrappers (DataParallel, torch.compile) add an
    # ``_orig_mod`` prefix to every parameter.  Remove it so loading succeeds.
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    log.success(f"Loaded '{path}'  (epoch {epoch},  vocab={len(vocab)},  params={model.num_parameters:,})")
    return model, vocab


def run_generation(
    model: CharRNN,
    vocab: Vocabulary,
    seed: str,
    length: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> None:
    output = _generate(
        model, vocab, seed,
        length=length,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    log.success("\n" + output + "\n")


def main() -> None:
    args = build_parser().parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.banner("shinrai — generate")
    model, vocab = load_model_from_checkpoint(args.checkpoint, device)

    if args.seed is not None:
        # Single generation and exit
        run_generation(
            model, vocab, args.seed,
            length=args.length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
    else:
        # Interactive mode
        log.info("Interactive mode — enter seed text (or 'exit' to quit).\n")
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
            run_generation(
                model, vocab, prompt,
                length=args.length,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )


if __name__ == "__main__":
    main()
