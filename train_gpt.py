import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
import os
import math
import matplotlib.pyplot as plt
from model_gpt import GPTSmall

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SEQ_LEN        = 50
BATCH_SIZE     = 64
EPOCHS         = 20
LR             = 3e-4
GRAD_CLIP      = 1.0
WARMUP_STEPS   = 200
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
    with open("tokenized_data.pkl", "rb") as f:
        tokenized_data = pickle.load(f)

    vocab_size, pad_idx = load_tokenizer_info()
    print(f"[INFO] Vocab size: {vocab_size} | Tokens: {len(tokenized_data)}")

    dataset    = TextDataset(tokenized_data, SEQ_LEN)
    # num_workers=0 — bezpieczne na każdej platformie (brak błędów forkserver)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model     = GPTSmall(vocab_size=vocab_size, seq_len=SEQ_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = (nn.CrossEntropyLoss(ignore_index=pad_idx)
                 if pad_idx >= 0 else nn.CrossEntropyLoss())

    total_steps = EPOCHS * len(dataloader)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    avg_losses = []
    best_loss  = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

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
        print(f"Epoch {epoch}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, f"gpt_epoch{epoch}.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "gpt_best.pt"))
            print(f"[BEST] loss={best_loss:.4f}")

    try:
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, EPOCHS + 1), avg_losses, marker="o", color="cyan")
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