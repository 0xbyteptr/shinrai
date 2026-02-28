"""
finetune_gpt.py — douczanie istniejącego modelu na nowych danych.

Użycie:
    py finetune_gpt.py --model checkpoints/gpt_best.pt --data new_tokenized.pkl
    py finetune_gpt.py --model checkpoints/gpt_best.pt --data new_tokenized.pkl --epochs 5 --lr 1e-4
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
import os
import math
import argparse
import matplotlib.pyplot as plt
from model_gpt import GPTSmall
from data_utils import load_tokens

# ──────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",   required=True,
                    help="Checkpoint bazowy do fine-tuningu")
parser.add_argument("--data",    default="tokenized_data.pkl",
                    help="New tokenized data file (.pkl or .bin)")
parser.add_argument("--tokenizer", default="tokenizer.pkl",
                    help="Tokenizer pasujący do checkpointu")
parser.add_argument("--epochs",  type=int,   default=5)
parser.add_argument("--lr",      type=float, default=1e-4,
                    help="Niższe LR niż przy treningu od zera (default: 1e-4)")
parser.add_argument("--batch",   type=int,   default=64)
parser.add_argument("--out_dir", default="checkpoints",
                    help="Gdzie zapisywać checkpointy")
parser.add_argument("--freeze_layers", type=int, default=0,
                    help="Ile warstw transformera zamrozić (0=trenuj wszystko)")
args = parser.parse_args()

SEQ_LEN   = 50
GRAD_CLIP = 1.0
os.makedirs(args.out_dir, exist_ok=True)

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx     : idx + self.seq_len],     dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def main():
    # ── Wczytaj dane ──────────────────────────
    # allow .bin or pickle token streams
    tokenized_data = load_tokens(args.data)
    print(f"[INFO] Nowe dane: {len(tokenized_data)} tokenów")

    # ── Odczytaj vocab_size z checkpointu ─────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model, map_location=device)
    vocab_size = state_dict["embed.weight"].shape[0]
    seq_len_ckpt = state_dict["pos_embed.weight"].shape[0]
    print(f"[INFO] Checkpoint: vocab={vocab_size}, seq_len={seq_len_ckpt} | Device: {device}")

    # ── Tokenizer (dla pad_idx) ───────────────
    pad_idx = -1
    if os.path.exists(args.tokenizer):
        with open(args.tokenizer, "rb") as f:
            raw = pickle.load(f)
        stoi    = raw if isinstance(raw, dict) else raw.stoi
        pad_idx = stoi.get("<PAD>", -1)

    # ── Model ─────────────────────────────────
    model = GPTSmall(vocab_size, seq_len=seq_len_ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    # Opcjonalne zamrożenie pierwszych N warstw
    if args.freeze_layers > 0:
        for i, layer in enumerate(model.transformer.layers):
            if i < args.freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        print(f"[INFO] Zamrożono {args.freeze_layers} warstw transformera")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametry: {trainable:,} / {total:,} trenowalnych")

    # ── DataLoader ────────────────────────────
    dataset    = TextDataset(tokenized_data, seq_len_ckpt)
    dataloader = DataLoader(dataset, batch_size=args.batch,
                            shuffle=True, num_workers=0, pin_memory=True)

    # ── Optimizer + Scheduler ─────────────────
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2
    )
    criterion = (nn.CrossEntropyLoss(ignore_index=pad_idx)
                 if pad_idx >= 0 else nn.CrossEntropyLoss())

    # Cosine annealing bez warmup (fine-tuning zaczyna od dobrego punktu)
    total_steps = args.epochs * len(dataloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Trening ───────────────────────────────
    avg_losses = []
    best_loss  = float("inf")
    base_name  = os.path.splitext(os.path.basename(args.model))[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"FT Epoch {epoch}/{args.epochs}", leave=False)

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / len(dataloader)
        avg_losses.append(avg_loss)
        print(f"FT Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        ckpt = os.path.join(args.out_dir, f"{base_name}_ft_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"[SAVED] {ckpt}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(args.out_dir, f"{base_name}_ft_best.pt")
            torch.save(model.state_dict(), best_ckpt)
            print(f"[BEST]  {best_ckpt} (loss={best_loss:.4f})")

    # ── Plot ──────────────────────────────────
    try:
        plt.figure(figsize=(8, 3))
        plt.plot(range(1, args.epochs + 1), avg_losses, marker="o", color="orange")
        plt.title(f"Fine-tuning Loss — {base_name}")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True); plt.tight_layout()
        out_png = f"{base_name}_ft_loss.png"
        plt.savefig(out_png)
        print(f"[INFO] Loss curve → {out_png}")
    except Exception as e:
        print(f"[WARNING] Plot error: {e}")

    print("[DONE] Fine-tuning complete")


if __name__ == "__main__":
    main()