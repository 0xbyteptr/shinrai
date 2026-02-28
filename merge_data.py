"""
merge_data.py — łączy wiele plików texts.pkl w jeden wspólny korpus
i buduje zunifikowany tokenizer.

Użycie:
    py merge_data.py --inputs texts1.pkl texts2.pkl texts3.pkl
    py merge_data.py --inputs texts1.pkl texts2.pkl --out merged_texts.pkl
"""
import pickle
import argparse
import os

def main(inputs, out="merged_texts.pkl"):
    """Merge several texts.pkl files into a single corpus.
    """
    merged = []
    for path in inputs:
        if not os.path.exists(path):
            print(f"[WARN] Nie znaleziono {path}, pomijam.")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[INFO] {path}: {len(data)} dokumentów")
        merged.extend(data)

    print(f"[INFO] Łącznie: {len(merged)} dokumentów")

    with open(out, "wb") as f:
        pickle.dump(merged, f)

    print(f"[DONE] Zapisano → {out}")
    print(f"[NEXT] Uruchom: py tokenize_texts.py --input {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Pliki texts.pkl do połączenia")
    parser.add_argument("--out", default="merged_texts.pkl",
                        help="Plik wyjściowy (default: merged_texts.pkl)")
    args = parser.parse_args()
    main(args.inputs, out=args.out)
