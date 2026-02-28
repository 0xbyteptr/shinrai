import pickle
from tokenizer_word import Tokenizer

# ── Wczytaj teksty ────────────────────────────────────────────────────
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

print(f"[INFO] Loaded {len(texts)} documents")

# ── Trenuj tokenizer ──────────────────────────────────────────────────
tokenizer = Tokenizer(min_freq=2)
tokenizer.fit(texts)
print(f"[INFO] Vocab size: {len(tokenizer)}")

# ── Tokenizuj wszystkie teksty ────────────────────────────────────────
tokenized_data = []
for text in texts:
    tokenized_data.extend(tokenizer.encode(text))

print(f"[INFO] Total tokens: {len(tokenized_data)}")

# ── Zapisz ────────────────────────────────────────────────────────────
with open("tokenized_data.pkl", "wb") as f:
    pickle.dump(tokenized_data, f)

tokenizer.save("tokenizer.pkl")

print("[DONE] tokenized_data.pkl and tokenizer.pkl saved")