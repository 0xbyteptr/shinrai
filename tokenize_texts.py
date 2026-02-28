"""
tokenize_texts.py — tokenizuje teksty i zapisuje tokenized_data.pkl + tokenizer.pkl

Użycie:
    py tokenize_texts.py                        # domyślnie texts.pkl
    py tokenize_texts.py --input merged_texts.pkl
"""
import pickle
import argparse
from tokenizer_word import Tokenizer
from data_utils import save_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="texts.pkl",
                    help="Plik z tekstami (default: texts.pkl)")
parser.add_argument("--out_tokens",    default="tokenized_data.pkl",
                    help="output token file (supports .pkl and .bin)")
parser.add_argument("--out_tokenizer", default="tokenizer.pkl")
args = parser.parse_args()


def main(input_file=None, out_tokens="tokenized_data.pkl", out_tokenizer="tokenizer.pkl"):
    """Tokenize texts and save token list + tokenizer."""
    input_file = input_file or args.input
    with open(input_file, "rb") as f:
        texts = pickle.load(f)

    print(f"[INFO] Loaded {len(texts)} documents from {input_file}")

    tokenizer = Tokenizer(min_freq=2)
    tokenizer.fit(texts)
    print(f"[INFO] Vocab size: {len(tokenizer)}")

    tokenized_data = []
    for text in texts:
        tokenized_data.extend(tokenizer.encode(text))

    print(f"[INFO] Total tokens: {len(tokenized_data)}")

    # save tokenized data using the helper (auto-detects extension)
    save_tokens(out_tokens, tokenized_data)

    tokenizer.save(out_tokenizer)
    print(f"[DONE] {out_tokens} + {out_tokenizer} saved")


if __name__ == "__main__":
    main(input_file=args.input,
         out_tokens=args.out_tokens,
         out_tokenizer=args.out_tokenizer)
