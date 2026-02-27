"""Training loop, checkpointing, and evaluation.

Encapsulates all stateful training logic in a single ``Trainer`` class so
that ``train.py`` remains a thin CLI entry point.

Features:
  - Sequence-level cross-entropy (predict all T chars per window)
  - Gradient clipping
  - AdamW + ReduceLROnPlateau scheduler
  - Train / validation split with perplexity logging
  - Early stopping with configurable patience
  - Periodic and best-model checkpointing
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from shinrai.config import CheckpointConfig, TrainConfig, Config
from shinrai.data import CharDataset, Vocabulary, acquire_text
from shinrai.logging import log
from shinrai.model import CharRNN

try:
    from tqdm import tqdm  # type: ignore[import]
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ──────────────────────────────────────────────────────────────────────────────
# Epoch metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_ppl: float
    val_loss: Optional[float]
    val_ppl: Optional[float]
    lr: float
    elapsed: float

    def __str__(self) -> str:
        parts = [
            f"Epoch {self.epoch:>3}",
            f"train_loss={self.train_loss:.4f}",
            f"ppl={self.train_ppl:.1f}",
        ]
        if self.val_loss is not None:
            parts += [f"val_loss={self.val_loss:.4f}", f"val_ppl={self.val_ppl:.1f}"]
        parts += [f"lr={self.lr:.2e}", f"{self.elapsed:.1f}s"]
        return "  ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: CharRNN,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    vocab: Vocabulary,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": model.config_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "chars": vocab.chars,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: CharRNN,
    optimizer: Optional[optim.Optimizer],
    scheduler,
    device: torch.device,
) -> tuple[int, list[str]]:
    """Load *path* into *model* (and optionally *optimizer*/*scheduler*).

    Returns:
        (epoch, chars)  — epoch the checkpoint was saved at and the vocab.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            log.warn("Could not restore optimizer state — architecture may differ.")

    if scheduler is not None and ckpt.get("scheduler_state"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass

    epoch = ckpt.get("epoch", 0)
    chars = ckpt.get("chars", [])
    log.success(f"Loaded checkpoint '{path}'  (epoch {epoch})")
    return epoch, chars


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Manages the full training lifecycle for a CharRNN model.

    Usage::

        trainer = Trainer(model, vocab, train_cfg, ckpt_cfg, device)
        trainer.load_checkpoint_if_needed()
        trainer.fit(encoded_data)
    """

    def __init__(
        self,
        model: CharRNN,
        vocab: Vocabulary,
        train_cfg: TrainConfig,
        ckpt_cfg: CheckpointConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.cfg = train_cfg
        self.ckpt = ckpt_cfg
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=train_cfg.lr_patience,
        )
        self.criterion = nn.CrossEntropyLoss()

        self._start_epoch = 0
        self._best_val_loss = float("inf")
        self._no_improve = 0

    # ── Checkpoint integration ────────────────────────────────────────────────

    def load_checkpoint_if_needed(self) -> None:
        path = self.ckpt.load_checkpoint
        if path is None and not self.ckpt.no_autoload:
            if self.ckpt.save_checkpoint and os.path.exists(self.ckpt.save_checkpoint):
                path = self.ckpt.save_checkpoint
                log.info(f"Auto-loading existing checkpoint: {path}")

        if path:
            try:
                epoch, chars = load_checkpoint(
                    path, self.model, self.optimizer, self.scheduler, self.device
                )
                self._start_epoch = epoch
                if chars and chars != self.vocab.chars:
                    log.warn(
                        "Vocabulary in checkpoint differs from current data. "
                        "Using checkpoint vocabulary."
                    )
                    self.vocab = Vocabulary.from_chars(chars)
            except Exception as exc:
                log.error(f"Failed to load checkpoint: {exc}")

    def _save(self, epoch: int, label: str = "") -> None:
        if not self.ckpt.save_checkpoint:
            return
        save_checkpoint(
            self.ckpt.save_checkpoint,
            self.model, self.optimizer, self.scheduler,
            epoch, self.vocab,
        )
        log.success(f"  Checkpoint saved: {self.ckpt.save_checkpoint}{label}")

    # ── Data loaders ──────────────────────────────────────────────────────────

    def _make_loaders(
        self, encoded: list[int], seq_length: int
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        full_ds = CharDataset(encoded, seq_length)
        if len(full_ds) == 0:
            raise RuntimeError("Dataset is empty — text is too short for seq_length.")

        val_size = int(len(full_ds) * self.cfg.val_split) if self.cfg.val_split > 0 else 0
        train_size = len(full_ds) - val_size

        if val_size > 0:
            train_ds, val_ds = random_split(full_ds, [train_size, val_size])
        else:
            train_ds, val_ds = full_ds, None

        # pin_memory=True with num_workers=0 can deadlock on Linux+CUDA;
        # only enable it when workers are actually spawning background threads.
        pin = torch.cuda.is_available() and self.device.type == "cuda"
        kw = dict(batch_size=self.cfg.batch_size, pin_memory=pin, num_workers=0)
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
        val_loader   = DataLoader(val_ds, shuffle=False, **kw) if val_ds else None

        log.info(
            f"Dataset: {len(train_ds):,} train samples"
            + (f", {len(val_ds):,} val samples" if val_ds else "")
        )
        return train_loader, val_loader

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits, _ = self.model(xb)
            B, T, V = logits.shape
            loss = self.criterion(logits.view(B * T, V), yb.view(B * T))
            total_loss  += loss.item() * (B * T)
            total_tokens += B * T
        return total_loss / max(total_tokens, 1)

    # ── One epoch of training ─────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss, total_tokens = 0.0, 0
        n_batches = len(loader)

        if _HAS_TQDM:
            iterator = tqdm(
                loader,
                desc=f"  Epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            iterator = loader

        for batch_idx, (xb, yb) in enumerate(iterator):
            if batch_idx == 0 and not _HAS_TQDM:
                print(f"  Epoch {epoch} — {n_batches} batches…", flush=True)
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()

            logits, _ = self.model(xb)
            B, T, V = logits.shape
            loss = self.criterion(logits.view(B * T, V), yb.view(B * T))
            loss.backward()

            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.optimizer.step()

            tokens = B * T
            total_loss  += loss.item() * tokens
            total_tokens += tokens

            # Fallback progress: print every 10% of batches when tqdm absent
            if not _HAS_TQDM and n_batches >= 10 and (batch_idx + 1) % max(1, n_batches // 10) == 0:
                pct = 100 * (batch_idx + 1) / n_batches
                avg = total_loss / total_tokens
                print(f"    [{pct:3.0f}%] batch {batch_idx+1}/{n_batches}  loss={avg:.4f}", flush=True)

        return total_loss / max(total_tokens, 1)

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(self, encoded: list[int], seq_length: int) -> list[EpochMetrics]:
        """Train for up to ``cfg.epochs`` epochs with early stopping.

        Periodic checkpointing is controlled by ``ckpt.save_every``; the
        default configuration (1) will write a checkpoint at the end of every
        epoch.  Set it to 0 to disable regular saves and rely solely on the
        best-model or final checkpointing behavior.  (The ``CheckpointConfig``
        default is already 1 so users get one checkpoint per epoch out of the
        box.)

        Args:
            encoded:    Full token sequence (integers) from the vocabulary.
            seq_length: Context window size (characters).

        Returns:
            List of :class:`EpochMetrics`, one per epoch.
        """
        train_loader, val_loader = self._make_loaders(encoded, seq_length)
        history: list[EpochMetrics] = []

        for epoch in range(self._start_epoch, self.cfg.epochs):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, epoch + 1)
            train_ppl  = math.exp(min(train_loss, 700))

            val_loss: Optional[float] = None
            val_ppl:  Optional[float] = None

            if val_loader:
                val_loss = self._evaluate(val_loader)
                val_ppl  = math.exp(min(val_loss, 700))
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)

            metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_ppl=train_ppl,
                val_loss=val_loss,
                val_ppl=val_ppl,
                lr=self.optimizer.param_groups[0]["lr"],
                elapsed=time.time() - t0,
            )
            history.append(metrics)
            log.print(str(metrics), style="bold" if val_loss else "")

            # Periodic checkpoint (default every epoch when save_every=1)
            # ``save_every`` may be overridden via CLI/Config.  A value of 0
            # disables periodic saving, leaving only best-model or end-of-run
            # checkpoints.
            if (
                self.ckpt.save_every > 0
                and (epoch + 1) % self.ckpt.save_every == 0
            ):
                self._save(epoch + 1, f" (epoch {epoch+1})")

            # Best-model checkpoint + early stopping
            if val_loss is not None:
                if val_loss < self._best_val_loss - 1e-4:
                    self._best_val_loss = val_loss
                    self._no_improve = 0
                    self._save(epoch + 1, " ← new best")
                else:
                    self._no_improve += 1
                    if self.cfg.patience > 0 and self._no_improve >= self.cfg.patience:
                        log.warn(
                            f"Early stopping: no improvement for {self._no_improve} epochs."
                        )
                        break

        # Final save when no val-based saving is active
        if not val_loader and self.ckpt.save_checkpoint:
            self._save(self.cfg.epochs)

        return history
# ──────────────────────────────────────────────────────────────────────────────
# High‑level convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

from typing import Optional


def _resolve_device(requested: Optional[str]) -> torch.device:
    """Choose a torch device string, warning on unavailable CUDA."""
    if requested:
        r = requested.lower()
        if r.startswith("cuda") and not torch.cuda.is_available():
            log.warn("CUDA requested but unavailable — falling back to CPU.")
            return torch.device("cpu")
        return torch.device(r)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(cfg: Config) -> tuple[CharRNN, Vocabulary]:
    """Execute the full training pipeline given a structured *cfg*.

    This mirrors the steps performed by the command‑line entry point but
    exposes a programmatic API usable by other scripts (for example, a remote
    trainer server).  It returns the trained model and vocabulary; the
    checkpoint path is taken from ``cfg.checkpoint.save_checkpoint``.
    """

    # ensure periodic checkpointing is sane
    if cfg.checkpoint.save_checkpoint and cfg.checkpoint.save_every <= 0:
        cfg.checkpoint.save_every = 1

    log.banner("shinrai — training")

    # text acquisition
    text = acquire_text(
        text_file=cfg.data.text_file,
        use_seed_articles=cfg.data.use_seed_articles,
        crawl=cfg.data.crawl,
        url=cfg.data.url,
        crawl_depth=cfg.data.crawl_depth,
        max_pages=cfg.data.max_pages,
        min_length=cfg.model.seq_length + 2,
    )
    log.info(f"Total characters: {len(text):,}")

    vocab = Vocabulary.from_text(text)
    log.info(f"Vocabulary size: {len(vocab)}")
    encoded = vocab.encode(text)

    device = _resolve_device(cfg.train.device)
    log.info(f"Device: {device}")

    model = CharRNN(
        vocab_size=len(vocab),
        embed_size=cfg.model.embed_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        cell_type=cfg.model.cell_type,
    ).to(device)
    log.info(f"Model parameters: {model.num_parameters:,}")

    trainer = Trainer(model, vocab, cfg.train, cfg.checkpoint, device)
    trainer.load_checkpoint_if_needed()
    trainer.fit(encoded, seq_length=cfg.model.seq_length)

    return model, vocab
