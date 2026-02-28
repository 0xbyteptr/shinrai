"""Centralised configuration using dataclasses.

All CLI arguments are parsed into one of these typed configs, which are then
passed down to the relevant subsystems.  This keeps argument validation
co-located with the data they describe and eliminates scattered ``args.*``
access throughout the codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Where and how to obtain training text.

    ``continue_from`` lets you load preprocessed data (and vocab) from a
    file saved by a previous run.  When present the other acquisition flags
    are ignored.

    ``crawl`` is now an integer representing the desired crawl depth.  A
    value of ``0`` (the default) disables crawling; ``1`` is the standard
    single‑page crawl.  The separate ``crawl_depth`` field is retained for
    backwards compatibility but is ignored if ``crawl`` is nonzero.
    """

    # Source selection (mutually exclusive)
    text_file: Optional[str] = None
    use_seed_articles: bool = False
    # crawl depth (0 = disabled).  Previously a bool flag.
    crawl: int = 0
    # concurrency settings to speed acquisition
    fetch_workers: int = 4     # parallel fetches for seed articles
    crawl_workers: int = 4     # parallel fetches during crawling
    # instead of fetching/encoding text, resume from a previously saved
    # data file (torch.save of ``{'encoded': [...], 'chars': [...]}'``)
    continue_from: Optional[str] = None

    url: str = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    crawl_depth: int = 1
    max_pages: int = 20


@dataclass
class ModelConfig:
    """RNN architecture hyper-parameters."""

    cell_type: Literal["lstm", "gru"] = "lstm"
    seq_length: int = 120
    embed_size: int = 128
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.3


@dataclass
class TrainConfig:
    """Training loop settings."""

    batch_size: int = 128
    epochs: int = 20
    lr: float = 0.002
    grad_clip: float = 5.0
    val_split: float = 0.05
    patience: int = 5          # early stopping (0 = disabled)
    lr_patience: int = 3       # ReduceLROnPlateau patience
    device: Optional[str] = None
    num_workers: int = 0        # DataLoader workers to accelerate loading
    use_amp: bool = False       # Enable automatic mixed precision (CUDA)
    use_compile: bool = False   # torch.compile (Dynamo) for potential speedups
    accumulate_steps: int = 1   # gradient accumulation to simulate larger batch
    data_parallel: bool = False # wrap model in nn.DataParallel if multiple GPUs


@dataclass
class CheckpointConfig:
    """Saving / loading model checkpoints."""

    save_checkpoint: Optional[str] = None
    # By default we save every epoch. A value of 0 disables periodic saves, in
    # which case only the best model (based on validation loss) or final model
    # will be written.
    save_every: int = 1        # save every N epochs (0 = val/best/end only)
    load_checkpoint: Optional[str] = None
    no_autoload: bool = False


@dataclass
class GenerateConfig:
    """Text generation settings."""

    temperature: float = 0.8
    top_p: float = 0.9
    gen_length: int = 300
    seed: str = "the"


# ──────────────────────────────────────────────────────────────────────────────
# Master config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)


# ──────────────────────────────────────────────────────────────────────────────
# Shared argument definitions (used by both train.py and generate.py)
# ──────────────────────────────────────────────────────────────────────────────

def add_data_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text_file", "--input_file", dest="text_file", metavar="PATH",
                     help="Local .txt or .pdf file to train on")
    src.add_argument("--use_seed_articles", action="store_true",
                     help="Fetch curated Wikipedia seed articles")
    # ``--crawl`` may be given without a value (defaults to depth=1) or
    # with an integer specifying depth.  argparse returns None when the flag
    # is absent, an int when provided, and 1 when written ``--crawl`` alone.
    src.add_argument("--crawl", nargs="?", const=1, type=int, metavar="DEPTH",
                     help="Crawl links starting from --url (optional depth)")
    p.add_argument("--fetch_workers", type=int, default=4,
                   help="Number of parallel downloads when fetching seed articles")
    p.add_argument("--crawl_workers", type=int, default=4,
                   help="Number of concurrent requests during crawling")
    p.add_argument("--continue_from", metavar="PATH",
                   help="Load preprocessed encoded data from a .pt file (skips acquisition).\n"
                   "If the given file is a training checkpoint, it will be treated"
                   "as --load_checkpoint instead and data will be acquired normally.")

    p.add_argument("--url",
                   default="https://en.wikipedia.org/wiki/Artificial_intelligence",
                   help="URL to fetch or crawl from")
    p.add_argument("--crawl_depth", type=int, default=1,
                   help="Crawler link-follow depth")
    p.add_argument("--max_pages", type=int, default=20,
                   help="Max pages to crawl")


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cell_type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--seq_length",  type=int,   default=120)
    p.add_argument("--embed_size",  type=int,   default=128)
    p.add_argument("--hidden_size", type=int,   default=512)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.3)


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=0.002)
    p.add_argument("--grad_clip",   type=float, default=5.0,
                   help="Max gradient norm (0 = disabled)")
    p.add_argument("--val_split",   type=float, default=0.05,
                   help="Fraction of data for validation")
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early-stopping patience (0 = disabled)")
    p.add_argument("--lr_patience", type=int,   default=3,
                   help="ReduceLROnPlateau patience")
    p.add_argument("--device",      default=None,
                   help="torch device string, e.g. 'cuda', 'cpu'")
    p.add_argument("--num_workers", type=int,   default=0,
                   help="Number of worker processes for data loading")
    p.add_argument("--use_amp",     action="store_true",
                   help="Enable automatic mixed precision (NVIDIA GPUs)")
    p.add_argument("--use_compile", action="store_true",
                   help="Run model through torch.compile() if available")
    p.add_argument("--accumulate_steps", type=int, default=1,
                   help="Accumulate gradients over this many batches")
    p.add_argument("--data_parallel", action="store_true",
                   help="Wrap model in DataParallel if multiple GPUs")


def add_checkpoint_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--save_checkpoint", default=None, metavar="PATH",
                   help="Where to save .pt checkpoints")
    p.add_argument("--save_every",  type=int, default=1,
                   help="Also save every N epochs (default 1 = each epoch)"
                   )
    p.add_argument("--load_checkpoint", default=None, metavar="PATH",
                   help="Resume from this checkpoint")
    p.add_argument("--no_autoload", action="store_true",
                   help="Skip auto-loading existing --save_checkpoint file")


def add_generate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p",       type=float, default=0.9,
                   help="Nucleus sampling p (1.0 = disabled)")
    p.add_argument("--gen_length",  type=int,   default=300,
                   help="Characters to generate")
    p.add_argument("--seed",        type=str,   default="the",
                   help="Seed text for generation (generate.py)")


# ──────────────────────────────────────────────────────────────────────────────
# Namespace → Config
# ──────────────────────────────────────────────────────────────────────────────

def config_from_namespace(args: argparse.Namespace) -> Config:
    """Convert a flat argparse Namespace into a structured Config."""

    def _get(attr, default=None):
        return getattr(args, attr, default)

    return Config(
        data=DataConfig(
            text_file=_get("text_file"),
            use_seed_articles=_get("use_seed_articles", False),
            crawl=_get("crawl", 0) or 0,
            fetch_workers=_get("fetch_workers", 4),
            crawl_workers=_get("crawl_workers", 4),
            continue_from=_get("continue_from"),
            url=_get("url", "https://en.wikipedia.org/wiki/Artificial_intelligence"),
            crawl_depth=_get("crawl_depth", 1),
            max_pages=_get("max_pages", 20),
        ),
        model=ModelConfig(
            cell_type=_get("cell_type", "lstm"),
            seq_length=_get("seq_length", 120),
            embed_size=_get("embed_size", 128),
            hidden_size=_get("hidden_size", 512),
            num_layers=_get("num_layers", 2),
            dropout=_get("dropout", 0.3),
        ),
        train=TrainConfig(
            batch_size=_get("batch_size", 128),
            epochs=_get("epochs", 20),
            lr=_get("lr", 0.002),
            grad_clip=_get("grad_clip", 5.0),
            val_split=_get("val_split", 0.05),
            patience=_get("patience", 5),
            lr_patience=_get("lr_patience", 3),
            device=_get("device"),
            num_workers=_get("num_workers", 0),
            use_amp=_get("use_amp", False),
            use_compile=_get("use_compile", False),
            accumulate_steps=_get("accumulate_steps", 1),
            data_parallel=_get("data_parallel", False),
        ),
        checkpoint=CheckpointConfig(
            save_checkpoint=_get("save_checkpoint"),
            save_every=_get("save_every", 1),
            load_checkpoint=_get("load_checkpoint"),
            no_autoload=_get("no_autoload", False),
        ),
        generate=GenerateConfig(
            temperature=_get("temperature", 0.8),
            top_p=_get("top_p", 0.9),
            gen_length=_get("gen_length", 300),
            seed=_get("seed", "the"),
        ),
    )
