import os
import glob
import sys

# ── ANSI colors ───────────────────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
C = "\033[96m"; M = "\033[95m"; B = "\033[1m"; X = "\033[0m"

LOGO = f"""{C}{B}
▄█████ ▄▄ ▄▄ ▄▄ ▄▄  ▄▄ ▄▄▄▄  ▄████▄ ██
▀▀▀▄▄▄ ██▄██ ██ ███▄██ ██▄█▄ ██▄▄██ ██
█████▀ ██ ██ ██ ██ ▀██ ██ ██ ██  ██ ██
{X}"""

MENU = f"""
  {C}1.{X} Build corpus      {Y}(scrape Wikipedia → texts.pkl){X}
  {C}2.{X} Tokenize texts    {Y}(texts.pkl → tokenized_data.pkl + tokenizer.pkl){X}
  {C}3.{X} Train GPT         {Y}(tokenized_data.pkl → checkpoints/){X}
  {C}4.{X} Chat with GPT     {Y}(wybierz checkpoint){X}
  {C}q.{X} Quit
"""


def pick_checkpoint() -> str:
    """Pozwól użytkownikowi wybrać checkpoint z listy."""
    checkpoints = sorted(glob.glob("checkpoints/*.pt"))
    if not checkpoints:
        print(f"{R}[ERROR]{X} Brak checkpointów w checkpoints/. Najpierw wytrenuj model.")
        return ""

    print(f"\n{Y}Dostępne checkpointy:{X}")
    for i, cp in enumerate(checkpoints, 1):
        print(f"  {C}{i}.{X} {cp}")
    print(f"  {C}{len(checkpoints)+1}.{X} {G}gpt_best.pt{X} (najlepszy epoch)")

    default = "checkpoints/gpt_best.pt" if os.path.exists("checkpoints/gpt_best.pt") \
              else checkpoints[-1]

    choice = input(f"{M}Wybierz (Enter = {default}): {X}").strip()

    if choice == "":
        return default
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            return checkpoints[idx]
    except ValueError:
        if os.path.exists(choice):
            return choice

    print(f"{Y}Nieprawidłowy wybór, używam {default}{X}")
    return default


def run(cmd: str):
    print(f"\n{G}[RUN]{X} {cmd}\n{'─'*50}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"{R}[ERROR]{X} Komenda zakończyła się kodem {ret}")


# ── MAIN ──────────────────────────────────────────────────────────────
print(LOGO)
print(f"{G}[INFO]{X} ShinrAI — GPT from scratch\n")

while True:
    print(MENU)
    choice = input(f"{M}>{X} ").strip().lower()

    match choice:
        case "1":
            run(f"{sys.executable} build_corpus.py")
        case "2":
            run(f"{sys.executable} tokenize_texts.py")
        case "3":
            run(f"{sys.executable} train_gpt.py")
        case "4":
            cp = pick_checkpoint()
            if cp:
                run(f"{sys.executable} chat_ui.py --model {cp}")
        case "q" | "quit" | "exit":
            print(f"\n{C}Bye!{X}")
            break
        case _:
            print(f"{R}Nieznana opcja.{X}")