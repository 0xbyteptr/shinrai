import torch
import torch.nn.functional as F
import pickle
import argparse
from tokenizer_word import Tokenizer
from model_gpt import GPTSmall

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SEQ_LEN     = 50
MAX_GEN     = 60
TEMPERATURE = 0.85   # > 1 bardziej losowe, < 1 bardziej deterministyczne
TOP_K       = 50     # zostaw tylko top-k tokenów
TOP_P       = 0.92   # nucleus sampling — zostaw tokeny z łącznym P >= top_p
REP_PENALTY = 1.3    # kara za powtarzanie tokenów

# ──────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ShinrAI ChatBot")
parser.add_argument("--model", default="checkpoints/gpt_best.pt",
                    help="Ścieżka do checkpointu modelu")
parser.add_argument("--temp",    type=float, default=TEMPERATURE)
parser.add_argument("--top_k",   type=int,   default=TOP_K)
parser.add_argument("--top_p",   type=float, default=TOP_P)
parser.add_argument("--max_gen", type=int,   default=MAX_GEN)
args = parser.parse_args()

# ──────────────────────────────────────────────
# TOKENIZER
# ──────────────────────────────────────────────
tokenizer = Tokenizer.load("tokenizer.pkl")
vocab_size = len(tokenizer)
print(f"[INFO] Vocab size: {vocab_size}")

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GPTSmall(vocab_size, seq_len=SEQ_LEN)
model.load_state_dict(torch.load(args.model, map_location=device))
model.to(device)
model.eval()
print(f"[INFO] Model loaded from {args.model} | Device: {device}")


# ──────────────────────────────────────────────
# SAMPLING HELPERS
# ──────────────────────────────────────────────
def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold  = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Usuń tokeny przekraczające próg p
    remove = cum_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float("-inf")
    # Przywróć oryginalną kolejność
    return logits.scatter(0, sorted_idx, sorted_logits)


# ──────────────────────────────────────────────
# GENERATE
# ──────────────────────────────────────────────
def generate(prompt: str,
             max_tokens: int  = args.max_gen,
             temperature: float = args.temp,
             top_k: int        = args.top_k,
             top_p: float      = args.top_p,
             rep_penalty: float = REP_PENALTY) -> str:

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.stoi.get("<UNK>", 1)]

    tokens    = tokens[-SEQ_LEN:]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(input_ids)[0, -1, :].clone()  # [vocab]

        # Repetition penalty
        for tid in set(generated):
            logits[tid] = (logits[tid] / rep_penalty if logits[tid] > 0
                           else logits[tid] * rep_penalty)

        # Temperature
        logits = logits / max(temperature, 1e-8)

        # Top-k + nucleus filtering
        logits = top_k_filter(logits, top_k)
        logits = top_p_filter(logits, top_p)

        probs      = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Stop przy <EOS>
        if next_token == tokenizer.stoi.get("<EOS>", -1):
            break

        generated.append(next_token)
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        input_ids   = torch.cat([input_ids, next_tensor], dim=1)[:, -SEQ_LEN:]

    return tokenizer.decode(generated)


# ──────────────────────────────────────────────
# ASCII UI
# ──────────────────────────────────────────────
W = 113  # szerokość ramki

def print_ui(user_text: str, bot_text: str, q_tokens: int, a_tokens: int):
    print("╔" + "═" * 46 + " ShinrAI Chat " + "═" * 46 + "╗")
    print(f"  Vocab: {vocab_size} | Seq Len: {SEQ_LEN} | "
          f"Temp: {args.temp} | Top-k: {args.top_k} | Top-p: {args.top_p}")
    print(f"  Q tokens: {q_tokens} | A tokens: {a_tokens}")
    print("╠" + "═" * W + "╣")
    # Zawijanie długich odpowiedzi
    for line in _wrap(f"You: {user_text}", W):
        print(f"  {line}")
    print("  " + "─" * (W - 2))
    for line in _wrap(f"Bot: {bot_text}", W):
        print(f"  {line}")
    print("╚" + "═" * W + "╝\n")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
CYAN, RESET = "\033[96m", "\033[0m"
print(f"\n{CYAN}[INFO]{RESET} ChatBot ready! Type 'exit' to quit.\n")

while True:
    try:
        user_input = input(f"{CYAN}You:{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Exiting.")
        break

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("[INFO] Bye!")
        break

    q_tokens   = len(tokenizer.encode(user_input))
    bot_output = generate(user_input)
    a_tokens   = len(tokenizer.encode(bot_output))
    print_ui(user_input, bot_output, q_tokens, a_tokens)