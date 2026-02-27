"""Character-level RNN model.

Supports both LSTM and GRU backends, multi-layer stacking, variational
dropout, and optional weight tying between the input embedding and the
output projection layer.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import torch
import torch.nn as nn

HiddenState = Union[
    tuple[torch.Tensor, torch.Tensor],  # LSTM: (h, c)
    torch.Tensor,                        # GRU:  h
]


class CharRNN(nn.Module):
    """Multi-layer character-level RNN with optional weight tying.

    Architecture:
        Embedding → Dropout → LSTM/GRU (N layers) → Dropout → Linear

    Weight tying: when ``embed_size == hidden_size`` the output projection
    shares its weight matrix with the input embedding, reducing parameter
    count and empirically improving perplexity.

    Args:
        vocab_size:  Number of unique characters.
        embed_size:  Embedding dimensionality.
        hidden_size: Hidden state size of each RNN layer.
        num_layers:  Number of stacked RNN layers.
        dropout:     Dropout probability applied after embedding and after RNN.
        cell_type:   ``"lstm"`` or ``"gru"``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: Literal["lstm", "gru"] = "lstm",
    ) -> None:
        super().__init__()

        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got '{cell_type}'")

        self.cell_type = cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.drop_in = nn.Dropout(dropout)

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # inter-layer dropout only makes sense with > 1 layer
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Weight tying (Press & Wolf, 2017)
        if embed_size == hidden_size:
            self.fc.weight = self.embed.weight

        self._init_weights()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 for LSTM (Jozefowicz et al., 2015)
                if self.cell_type == "lstm":
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[HiddenState] = None,
    ) -> tuple[torch.Tensor, HiddenState]:
        """
        Args:
            x:      LongTensor of shape ``(batch, seq_len)``.
            hidden: Optional initial hidden state.

        Returns:
            logits: FloatTensor of shape ``(batch, seq_len, vocab_size)``.
            hidden: Updated hidden state.
        """
        emb = self.drop_in(self.embed(x))      # (B, T, E)
        out, hidden = self.rnn(emb, hidden)     # (B, T, H)
        out = self.drop_out(out)
        logits = self.fc(out)                   # (B, T, V)
        return logits, hidden

    # ── Utilities ────────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> HiddenState:
        """Return a zero-initialised hidden state for the given batch size."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.cell_type == "lstm":
            return (h, torch.zeros_like(h))
        return h

    def detach_hidden(self, hidden: HiddenState) -> HiddenState:
        """Detach hidden state from the computation graph (TBPTT)."""
        if isinstance(hidden, tuple):
            return tuple(h.detach() for h in hidden)  # type: ignore[return-value]
        return hidden.detach()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def config_dict(self) -> dict:
        """Return a plain dict sufficient to recreate this model."""
        return {
            "vocab_size":  self.vocab_size,
            "embed_size":  self.embed.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "cell_type":   self.cell_type,
            # dropout is not stored in the model state but we stash it anyway
        }

    @classmethod
    def from_config(cls, cfg: dict, dropout: float = 0.0) -> "CharRNN":
        return cls(dropout=dropout, **cfg)
