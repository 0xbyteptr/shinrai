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
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_config": model.config_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "chars": vocab.chars,
    }
    if scaler is not None:
        data["scaler_state"] = scaler.state_dict()
    torch.save(data, path)


def load_checkpoint(
    path: str,
    model: CharRNN,
    optimizer: Optional[optim.Optimizer],
    scheduler,
    device: torch.device,
) -> tuple[int, list[str], Optional[dict]]:
    """Load *path* into *model* (and optionally *optimizer*/*scheduler*).

    The checkpoint may have been saved with a different vocabulary than the
    one used to construct *model*.  Previously we let :func:`load_state_dict`
    raise an exception in that case; now we catch mismatches and fall back to
    a non-strict load so that compatible parameters are restored and the
    remaining weights (e.g. embedding/output layers) stay at their
    initialized values.

    Returns:
        (epoch, chars, scaler_state)  — epoch the checkpoint was saved at,
        the vocab, and (if present) the AMP scaler state.
    """
    ckpt = torch.load(path, map_location=device)

    # try a strict load first so we catch unexpected missing keys, but don't
    # crash if the shapes don't line up (vocab changed, etc.).
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        log.warn(f"Model state dict mismatch: {exc}. loading partial weights.")
        model.load_state_dict(ckpt["model_state"], strict=False)

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

    # AMP scaler may be restored by caller if they have one
    scaler_state = ckpt.get("scaler_state")
    if scaler_state is not None:
        try:
            return epoch, chars, scaler_state
        except Exception:
            pass

    epoch = ckpt.get("epoch", 0)
    chars = ckpt.get("chars", [])
    scaler_state = ckpt.get("scaler_state")
    log.success(f"Loaded checkpoint '{path}'  (epoch {epoch})")
    return epoch, chars, scaler_state


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

        # optionally wrap in DataParallel if user requests and multiple GPUs
        if self.cfg.data_parallel and torch.cuda.device_count() > 1:
            log.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # compile model if requested (PyTorch 2.0).  Triton is required for
        # most backends; if it isn't installed the compiled model will raise an
        # error on first call, so we proactively check by running a tiny
        # inference and falling back if anything goes wrong.
        if self.cfg.use_compile and hasattr(torch, "compile"):
            try:
                compiled = torch.compile(self.model)
                # sanity check: run a tiny dummy forward to trigger backend
                try:
                    dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    _ = compiled(dummy)
                    self.model = compiled
                    log.info("Model compiled with torch.compile()")
                except Exception as e2:
                    log.warn(f"Compiled model failed at test run ({e2}); "
                             "falling back to uncompiled model.")
            except Exception as exc:
                log.warn(f"Failed to compile model: {exc}")

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

        # AMP scaler if requested and running on CUDA
        self.use_amp = train_cfg.use_amp and torch.cuda.is_available()
        self.scaler: Optional[torch.amp.GradScaler]
        if self.use_amp:
            # simply create scaler; latest PyTorch auto-detects device
            try:
                self.scaler = torch.amp.GradScaler()
            except TypeError:
                # very old versions might not have amp; fall back to None
                self.scaler = None
        else:
            self.scaler = None

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
                epoch, chars, scaler_state = load_checkpoint(
                    path, self.model, self.optimizer, self.scheduler, self.device
                )
                self._start_epoch = epoch
                if chars and chars != self.vocab.chars:
                    log.warn(
                        "Vocabulary in checkpoint differs from current data. "
                        "Using checkpoint vocabulary."
                    )
                    self.vocab = Vocabulary.from_chars(chars)
                if scaler_state and self.scaler is not None:
                    try:
                        self.scaler.load_state_dict(scaler_state)
                    except Exception:
                        pass
            except Exception as exc:
                log.error(f"Failed to load checkpoint: {exc}")

    def _save(self, epoch: int, label: str = "") -> None:
        if not self.ckpt.save_checkpoint:
            return
        save_checkpoint(
            self.ckpt.save_checkpoint,
            self.model, self.optimizer, self.scheduler,
            epoch, self.vocab,
            self.scaler if self.use_amp else None,
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

        # pin_memory helps when using CUDA; only enable if workers are
        # spawning threads for loading.
        pin = torch.cuda.is_available() and self.device.type == "cuda"
        num_workers = max(0, self.cfg.num_workers)
        kw = dict(batch_size=self.cfg.batch_size, pin_memory=pin,
                  num_workers=num_workers, persistent_workers=num_workers>0)
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
        # enable cuDNN autotuner for potential speed-ups on fixed input
        torch.backends.cudnn.benchmark = True
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

        self.optimizer.zero_grad()
        for batch_idx, (xb, yb) in enumerate(iterator):
            step_idx = batch_idx + 1
            if batch_idx == 0 and not _HAS_TQDM:
                print(f"  Epoch {epoch} — {n_batches} batches…", flush=True)
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)

            # forward / backward inside amp context if enabled
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits, _ = self.model(xb)
                    B, T, V = logits.shape
                    loss = self.criterion(logits.view(B * T, V), yb.view(B * T))
                self.scaler.scale(loss).backward()
            else:
                logits, _ = self.model(xb)
                B, T, V = logits.shape
                loss = self.criterion(logits.view(B * T, V), yb.view(B * T))
                loss.backward()

            # accumulate gradients
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            if step_idx % self.cfg.accumulate_steps == 0:
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

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

    # text acquisition / loading
    if cfg.data.continue_from:
        log.info(f"Loading preprocessed data from {cfg.data.continue_from}")
        ck = torch.load(cfg.data.continue_from, map_location="cpu")
        # two possible valid formats: a data dump with 'encoded'/'chars', or a
        # full training checkpoint.  The latter is commonly mistaken for the
        # former, so we detect it and redirect the path into load_checkpoint.
        if "encoded" in ck and "chars" in ck:
            encoded = ck.get("encoded")
            chars = ck.get("chars")
            vocab = Vocabulary.from_chars(chars)
            log.info(f"Loaded encoded data: {len(encoded):,} tokens, vocab={len(vocab)}")
        elif "model_state" in ck:
            log.warn("continue_from refers to a model checkpoint; "
                     "treating it as --load_checkpoint instead.")
            # set the checkpoint path and fall through to acquisition below
            cfg.checkpoint.load_checkpoint = cfg.data.continue_from
            cfg.data.continue_from = None
            crawl_depth = cfg.data.crawl or cfg.data.crawl_depth
            text = acquire_text(
                text_file=cfg.data.text_file,
                use_seed_articles=cfg.data.use_seed_articles,
                crawl=crawl_depth > 0,
                url=cfg.data.url,
                crawl_depth=crawl_depth,
                max_pages=cfg.data.max_pages,
                min_length=cfg.model.seq_length + 2,
            )
            log.info(f"Total characters: {len(text):,}")
            vocab = Vocabulary.from_text(text)
            log.info(f"Vocabulary size: {len(vocab)}")
            encoded = vocab.encode(text)
        else:
            raise RuntimeError("continue_from file missing 'encoded' or 'chars' keys")
    else:
        crawl_depth = cfg.data.crawl or cfg.data.crawl_depth
        text = acquire_text(
            text_file=cfg.data.text_file,
            use_seed_articles=cfg.data.use_seed_articles,
            crawl=crawl_depth > 0,
            url=cfg.data.url,
            crawl_depth=crawl_depth,
            max_pages=cfg.data.max_pages,
            min_length=cfg.model.seq_length + 2,
            fetch_workers=cfg.data.fetch_workers,
            crawl_workers=cfg.data.crawl_workers,
        )
        log.info(f"Total characters: {len(text):,}")

        vocab = Vocabulary.from_text(text)
        log.info(f"Vocabulary size: {len(vocab)}")
        encoded = vocab.encode(text)

    # If we're going to load a training checkpoint, its vocabulary may
    # differ from the text we just acquired.  Earlier versions built the
    # model before inspecting the checkpoint, which meant load_state_dict()
    # could throw due to mismatched embedding/output dimensions.  To avoid
    # that, check for a saved vocab now and (if necessary) override the
    # vocabulary and re-encode the text so that the model we construct
    # below will match the checkpoint exactly.
    if cfg.checkpoint.load_checkpoint:
        try:
            ckpt = torch.load(cfg.checkpoint.load_checkpoint, map_location="cpu")
            ck_chars = ckpt.get("chars")
            if ck_chars:
                if ck_chars != vocab.chars:
                    log.warn(
                        "Vocabulary in checkpoint differs from current data. "
                        "Using checkpoint vocabulary."
                    )
                    vocab = Vocabulary.from_chars(ck_chars)
                    # re-encode using checkpoint vocab; unknown chars map to space
                    encoded = vocab.encode(text)
        except Exception:
            # silently ignore issues reading the checkpoint; they will be
            # surfaced later during the actual load.
            pass

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
