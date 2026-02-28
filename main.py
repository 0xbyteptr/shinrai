import os
import glob
import sys
import argparse
import subprocess

# terminal colours
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
C = "\033[96m"; M = "\033[95m"; B = "\033[1m"; X = "\033[0m"

LOGO = f"""{C}{B}
▄█████ ▄▄ ▄▄ ▄▄ ▄▄  ▄▄ ▄▄▄▄  ▄████▄ ██
▀▀▀▄▄▄ ██▄██ ██ ███▄██ ██▄█▄ ██▄▄██ ██
█████▀ ██ ██ ██ ██ ▀██ ██ ██ ██  ██ ██
{X}"""

MENU = f"""   ### interactive menu ###
  {C}1.{X} Build corpus       {Y}(scrape Wikipedia → texts.pkl){X}
  {C}2.{X} Merge corpora      {Y}(połącz wiele texts.pkl w jeden){X}
  {C}3.{X} Tokenize texts     {Y}(texts.pkl → tokenized_data.pkl/.bin + tokenizer.pkl){X}
  {C}4.{X} Train GPT          {Y}(tokenized_data.pkl/.bin → checkpoints/){X}
  {C}5.{X} Fine-tune GPT      {Y}(douczaj istniejący model na nowych danych, token file may be .pkl or .bin){X}
  {C}6.{X} Chat with GPT      {Y}(wybierz checkpoint){X}
  {C}q.{X} Quit
"""


def pick_checkpoint(label="model") -> str:
    checkpoints = sorted(glob.glob("checkpoints/*.pt"))
    if not checkpoints:
        print(f"{R}[ERROR]{X} Brak checkpointów. Najpierw wytrenuj model.")
        return ""
    print(f"\n{Y}Dostępne checkpointy:{X}")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {C}{i}.{X} {cp}")
    default = "checkpoints/gpt_best.pt" if os.path.exists("checkpoints/gpt_best.pt") \
              else checkpoints[-1]
    choice = input(f"{M}Wybierz {label} (Enter = {default}): {X}").strip()
    if choice == "":
        return default
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            return checkpoints[idx]
    except ValueError:
        if os.path.exists(choice):
            return choice
    return default


def pick_pkl(prompt, default) -> str:
    # generic picker for any data file (used mainly for tokenized data);
    # we retain the old name for backwards compatibility but the prompt
    # now mentions .pkl or .bin
    val = input(f"{M}{prompt} (Enter = {default}): {X}").strip()
    return val if val else default


def run(cmd: str):
    print(f"\n{G}[RUN]{X} {cmd}\n{'─'*50}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"{R}[ERROR]{X} Komenda zakończyła się kodem {ret}")


PY = sys.executable

# === command line interface ===
cli_parser = argparse.ArgumentParser(description="ShinrAI workflow manager")
subparsers = cli_parser.add_subparsers(dest="cmd", help="sub-command to run")

# build corpus
p = subparsers.add_parser("build", help="scrape Wikipedia and save texts")
p.add_argument("--start", nargs="+", default=["https://en.wikipedia.org/"],
               help="starting URLs")
p.add_argument("--limit", type=int, default=20, help="max pages to scrape")
p.add_argument("--out", default="texts.pkl", help="output file")

# merge corpora
p = subparsers.add_parser("merge", help="merge several texts.pkl files")
p.add_argument("inputs", nargs="+", help="input pickle files")
p.add_argument("--out", default="merged_texts.pkl", help="output file")

# tokenize
p = subparsers.add_parser("tokenize", help="tokenize text corpus")
p.add_argument("--input", default="texts.pkl", help="input texts pickle")
p.add_argument("--out_tokens", default="tokenized_data.pkl")
p.add_argument("--out_tokenizer", default="tokenizer.pkl")

# other commands just forward
subparsers.add_parser("train", help="train a new GPT model")
subparsers.add_parser("finetune", help="fine-tune an existing GPT model")
subparsers.add_parser("chat", help="launch interactive chat using a checkpoint")

cli_args, leftovers = cli_parser.parse_known_args()

if cli_args.cmd:
    # run non-interactive command and exit
    if cli_args.cmd == "build":
        run(f"{PY} build_corpus.py --start {' '.join(cli_args.start)} "
            f"--limit {cli_args.limit} --out {cli_args.out}")
    elif cli_args.cmd == "merge":
        run(f"{PY} merge_data.py --inputs {' '.join(cli_args.inputs)} "
            f"--out {cli_args.out}")
    elif cli_args.cmd == "tokenize":
        run(f"{PY} tokenize_texts.py --input {cli_args.input} "
            f"--out_tokens {cli_args.out_tokens} "
            f"--out_tokenizer {cli_args.out_tokenizer}")
    elif cli_args.cmd in ("train", "finetune", "chat"):
        script = {
            "train": "train_gpt.py",
            "finetune": "finetune_gpt.py",
            "chat": "chat_ui.py",
        }[cli_args.cmd]
        run(f"{PY} {script} {' '.join(leftovers)}")
    sys.exit(0)

print(LOGO)
print(f"{G}[INFO]{X} ShinrAI — GPT from scratch\n")

while True:
    print(MENU)
    choice = input(f"{M}>{X} ").strip().lower()

    match choice:
        case "1":
            run(f"{PY} build_corpus.py")

        case "2":
            # Merge wielu texts.pkl
            print(f"\n{Y}Podaj ścieżki do plików texts.pkl (spacja między nimi):{X}")
            inputs = input(f"{M}Pliki: {X}").strip()
            if not inputs:
                print(f"{R}Nie podano plików.{X}")
                continue
            out = pick_pkl("Plik wyjściowy", "merged_texts.pkl")
            run(f"{PY} merge_data.py --inputs {inputs} --out {out}")
            # Od razu zaproponuj tokenizację
            tok = input(f"\n{M}Tokenizować teraz? [t/N]: {X}").strip().lower()
            if tok == "t":
                run(f"{PY} tokenize_texts.py --input {out}")

        case "3":
            src = pick_pkl("Plik z tekstami", "texts.pkl")
            run(f"{PY} tokenize_texts.py --input {src}")

        case "4":
            run(f"{PY} train_gpt.py")

        case "5":
            # Fine-tuning
            print(f"\n{Y}=== Fine-tuning ==={X}")
            base_model = pick_checkpoint("model bazowy")
            if not base_model:
                continue
            data = pick_pkl("Nowe dane tokenized (pkl/.bin)", "tokenized_data.pkl")
            epochs = input(f"{M}Epoki fine-tuningu (Enter = 5): {X}").strip()
            epochs = epochs if epochs.isdigit() else "5"
            lr     = input(f"{M}Learning rate (Enter = 1e-4): {X}").strip()
            lr     = lr if lr else "1e-4"
            freeze = input(f"{M}Zamrożone warstwy transformera (Enter = 0): {X}").strip()
            freeze = freeze if freeze.isdigit() else "0"
            run(f"{PY} finetune_gpt.py "
                f"--model {base_model} --data {data} "
                f"--epochs {epochs} --lr {lr} --freeze_layers {freeze}")

        case "6":
            cp = pick_checkpoint()
            if cp:
                run(f"{PY} chat_ui.py --model {cp}")

        case "q" | "quit" | "exit":
            print(f"\n{C}Bye!{X}")
            break

        case _:
            print(f"{R}Nieznana opcja.{X}")