import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse

# ──────────────────────────────────────────────
# LEGACY MODEL — dokładna kopia oryginalnej architektury
# ──────────────────────────────────────────────
class GPTSmall(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=4, num_layers=6,
                 seq_len=100, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed     = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(seq_len, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        b, l = x.size()
        pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        x = self.embed(x) + self.pos_embed(pos)
        x = self.transformer(x)
        x = self.ln(x)
        return self.fc(x)


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SEQ_LEN     = 50
TEMPERATURE = 0.85
TOP_K       = 50
TOP_P       = 0.92
REP_PENALTY = 1.3

parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="checkpoints/gpt_epoch20.pt")
parser.add_argument("--temp",    type=float, default=TEMPERATURE)
parser.add_argument("--top_k",   type=int,   default=TOP_K)
parser.add_argument("--top_p",   type=float, default=TOP_P)
parser.add_argument("--max_gen", type=int,   default=60)
args = parser.parse_args()

# ──────────────────────────────────────────────
# WCZYTAJ vocab_size Z CHECKPOINTU (nie z tokenizera!)
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(args.model, map_location=device)
vocab_size  = state_dict["embed.weight"].shape[0]
seq_len_ckpt = state_dict["pos_embed.weight"].shape[0]
print(f"[INFO] Checkpoint vocab_size={vocab_size}, seq_len={seq_len_ckpt}")

# ──────────────────────────────────────────────
# TOKENIZER — dopasuj do vocab_size z checkpointu
# ──────────────────────────────────────────────
with open("tokenizer.pkl", "rb") as f:
    raw = pickle.load(f)

stoi = raw if isinstance(raw, dict) else raw.stoi

# Jeśli vocab checkpointu != aktualny tokenizer — ostrzeż użytkownika
if len(stoi) != vocab_size:
    print(f"[WARN] tokenizer.pkl ma {len(stoi)} tokenów, "
          f"ale checkpoint oczekuje {vocab_size}.")
    print("[WARN] Upewnij się że używasz właściwego tokenizer.pkl "
          "dla tego checkpointu.")

itos = {v: k for k, v in stoi.items()}

class SimpleTokenizer:
    def encode(self, text):
        return [stoi[w] for w in text.lower().split() if w in stoi]
    def decode(self, tokens):
        return " ".join(itos.get(t, "<UNK>") for t in tokens)

tokenizer = SimpleTokenizer()

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
model = GPTSmall(vocab_size, seq_len=seq_len_ckpt)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[INFO] Model loaded from {args.model} | Device: {device}")


# ──────────────────────────────────────────────
# SAMPLING
# ──────────────────────────────────────────────
def top_k_filter(logits, k):
    if k <= 0: return logits
    vals, _ = torch.topk(logits, k)
    return logits.masked_fill(logits < vals[-1], float("-inf"))

def top_p_filter(logits, p):
    if p >= 1.0: return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float("-inf")
    return logits.scatter(0, sorted_idx, sorted_logits)

def generate(prompt):
    tokens = tokenizer.encode(prompt)
    if not tokens: tokens = [0]
    tokens    = tokens[-seq_len_ckpt:]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = []

    for _ in range(args.max_gen):
        with torch.no_grad():
            logits = model(input_ids)[0, -1, :].clone()

        for tid in set(generated):
            logits[tid] = logits[tid] / REP_PENALTY if logits[tid] > 0 \
                          else logits[tid] * REP_PENALTY

        logits = logits / max(args.temp, 1e-8)
        logits = top_k_filter(logits, args.top_k)
        logits = top_p_filter(logits, args.top_p)
        probs  = F.softmax(logits, dim=-1)
        next_t = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_t)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_t]], dtype=torch.long, device=device)], dim=1
        )[:, -seq_len_ckpt:]

    return tokenizer.decode(generated)


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
W = 110

def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    return lines + [cur] if cur else lines

def print_ui(user, bot, qt, at):
    print("╔" + "═"*46 + " ShinrAI Legacy " + "═"*44 + "╗")
    print(f"  Vocab: {vocab_size} | Seq: {seq_len_ckpt} | Temp: {args.temp} | Top-k: {args.top_k} | Top-p: {args.top_p}")
    print(f"  Q tokens: {qt} | A tokens: {at}")
    print("╠" + "═"*W + "╣")
    for line in wrap(f"You: {user}", W): print(f"  {line}")
    print("  " + "─"*(W-2))
    for line in wrap(f"Bot: {bot}",  W): print(f"  {line}")
    print("╚" + "═"*W + "╝\n")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
CYAN, RESET = "\033[96m", "\033[0m"
print(f"\n{CYAN}[INFO]{RESET} Legacy ChatBot ready! Type 'exit' to quit.\n")

while True:
    try:
        user_input = input(f"{CYAN}You:{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Exiting."); break
    if not user_input: continue
    if user_input.lower() in ("exit", "quit"):
        print("[INFO] Bye!"); break

    qt  = len(tokenizer.encode(user_input))
    out = generate(user_input)
    at  = len(tokenizer.encode(out))
    print_ui(user_input, out, qt, at)