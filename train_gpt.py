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
# DEFAULT CONFIG (can be overridden by CLI args)
# ──────────────────────────────────────────────
DEFAULT_SEQ_LEN        = 100
DEFAULT_BATCH_SIZE     = 32
DEFAULT_EPOCHS         = 20
DEFAULT_LR             = 3e-4
DEFAULT_GRAD_CLIP      = 1.0
DEFAULT_WARMUP_STEPS   = 200
DEFAULT_CHECKPOINT_DIR = "checkpoints"

os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, tokenized_data, seq_len):
        self.data    = tokenized_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx     : idx + self.seq_len],     dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def load_tokenizer_info():
    """Obsługuje nowy Tokenizer i stary dict {word: idx}."""
    with open("tokenizer.pkl", "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        return len(raw), -1          # brak <PAD> w starym formacie
    return len(raw.stoi), raw.stoi.get("<PAD>", -1)


# ──────────────────────────────────────────────
# MAIN  ← wymagane przez multiprocessing/forkserver
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train GPT model from scratch.")
    parser.add_argument("--tokenized", default="tokenized_data.pkl",
                        help="Tokenized data file (.pkl or .bin) to train on")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--out_dir", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    # support both pickle and binary token list
    tokenized_data = load_tokens(args.tokenized)

    vocab_size, pad_idx = load_tokenizer_info()
    print(f"[INFO] Vocab size: {vocab_size} | Tokens: {len(tokenized_data)}")

    # make sure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    dataset    = TextDataset(tokenized_data, args.seq_len)
    # num_workers=0 — bezpieczne na każdej platformie (brak błędów forkserver)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                            num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model     = GPTSmall(vocab_size=vocab_size, seq_len=args.seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = (nn.CrossEntropyLoss(ignore_index=pad_idx)
                 if pad_idx >= 0 else nn.CrossEntropyLoss())

    total_steps = args.epochs * len(dataloader)

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    avg_losses = []
    best_loss  = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / len(dataloader)
        avg_losses.append(avg_loss)
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, f"gpt_epoch{epoch}.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, "gpt_best.pt"))
            print(f"[BEST] loss={best_loss:.4f}")

    try:
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, args.epochs + 1), avg_losses, marker="o", color="cyan")
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig("loss_curve.png")
        plt.show()
        print("[INFO] Loss curve saved to loss_curve.png")
    except Exception as e:
        print(f"[WARNING] Could not show plot: {e}")

    print("[DONE] Training complete")


if __name__ == "__main__":
    main()