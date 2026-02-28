import torch
import torch.nn as nn
import math


class GPTSmall(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=4, num_layers=6,
                 seq_len=100, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(seq_len, embed_size)
        self.drop = nn.Dropout(dropout)

        # Causal (decoder-style) transformer — każdy token widzi tylko wcześniejsze
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN — stabilniejszy trening
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size, bias=False)

        # Weight tying — embedding i output dzielą wagi (mniejszy model, lepsza generalizacja)
        self.fc.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, size, device):
        """Maska autoregresywna: token i widzi tylko tokeny 0..i"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        """
        x:   [batch, seq_len]  — tokeny wejściowe
        out: [batch, seq_len, vocab_size]
        """
        b, l = x.size()
        positions = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)

        emb = self.drop(self.embed(x) + self.pos_embed(positions))

        causal_mask = self._causal_mask(l, x.device)

        # TransformerDecoder z memory=emb (self-attention z causal mask)
        out = self.transformer(emb, emb, tgt_mask=causal_mask, memory_mask=causal_mask)
        out = self.ln(out)
        return self.fc(out)