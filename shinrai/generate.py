"""Text generation / sampling utilities.

Provides nucleus (top-p) sampling with temperature scaling, which produces
more coherent text than pure temperature sampling by restricting the
probability mass to the most likely tokens.

Reference: Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shinrai.model import CharRNN
from shinrai.data import Vocabulary


# ──────────────────────────────────────────────────────────────────────────────
# Sampling primitives
# ──────────────────────────────────────────────────────────────────────────────

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature (higher = more random, lower = sharper)."""
    return logits / max(temperature, 1e-8)


def _nucleus_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out tokens outside the top-p probability mass.

    Args:
        probs:  1-D probability tensor (already softmax'd).
        top_p:  Cumulative probability threshold in (0, 1].

    Returns:
        Re-normalised probability tensor with low-probability tokens zeroed.
    """
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens once the cumulative mass *before* this token exceeds top_p
    remove_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-8)

    # Scatter back to original indices
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_idx, sorted_probs)
    return filtered


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> int:
    """Sample one token index from a logit vector.

    Args:
        logits:      1-D float tensor of shape ``(vocab_size,)``.
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.

    Returns:
        Sampled token index (int).
    """
    logits = _apply_temperature(logits, temperature)
    probs = F.softmax(logits, dim=-1)
    probs = _nucleus_filter(probs, top_p)
    return torch.multinomial(probs, num_samples=1).item()  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# High-level generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: CharRNN,
    vocab: Vocabulary,
    seed_str: str,
    length: int = 300,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seq_length: int = 120,
    device: torch.device | None = None,
) -> str:
    """Autoregressively generate ``length`` characters from ``seed_str``.

    Args:
        model:       Trained CharRNN (will be set to eval mode internally).
        vocab:       Vocabulary corresponding to the trained model.
        seed_str:    Prime text; characters not in vocab are replaced with space.
        length:      Number of new characters to generate.
        temperature: Sampling temperature (higher = more creative).
        top_p:       Nucleus sampling threshold (1.0 disables it).
        seq_length:  Context window the model was trained with.
        device:      Inference device (defaults to model's current device).

    Returns:
        seed_str + generated continuation as a single string.
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Encode and pad / trim seed to exactly seq_length
    seed_ids = vocab.encode(seed_str.lower())
    if len(seed_ids) < seq_length:
        pad_id = vocab.char2idx.get(" ", 0)
        seed_ids = [pad_id] * (seq_length - len(seed_ids)) + seed_ids
    else:
        seed_ids = seed_ids[-seq_length:]

    # Warm up the hidden state with the full seed
    inp = torch.tensor([seed_ids], dtype=torch.long, device=device)
    _, hidden = model(inp)

    # Autoregressively sample one token at a time
    current_id = seed_ids[-1]
    generated_ids: list[int] = []

    for _ in range(length):
        inp = torch.tensor([[current_id]], dtype=torch.long, device=device)
        logits, hidden = model(inp, hidden)
        current_id = sample_token(logits[0, 0], temperature=temperature, top_p=top_p)
        generated_ids.append(current_id)

    return seed_str + vocab.decode(generated_ids)
