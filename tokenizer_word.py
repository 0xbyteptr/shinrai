import re
from collections import Counter


class Tokenizer:
    SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, min_freq: int = 2):
        """
        min_freq: słowa rzadsze niż min_freq zostaną zastąpione przez <UNK>
        """
        self.min_freq = min_freq
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenizacja na poziomie słów.

        Pojedyncze tokeny obejmują słowa oraz znaki interpunkcyjne, ale
        apostrofy wewnątrz słów nie są rozdzielane (dzięki temu
        contractions takie jak "you'll" albo "don't" pozostają jednym
        tokenem).  Stare regexy dzieliły apostrof na osobny token, co
        prowadziło do dziwnych wyjść typu "you' ll".
        """
        # 
        # r"\w+(?:'\w+)*" → słowo (alfanumeryczne) opcjonalnie z kilkoma
        # segmentami zaczynającymi się apostrofem, np. don't, o'clock
        # r"[^\s\w]"           → każdy inny nie-białoskracowy, nie-znak
        #                    alfanumeryczny (czyli większość interpunkcji)
        return re.findall(r"\w+(?:'\w+)*|[^\s\w]", text.lower())

    # ------------------------------------------------------------------
    def fit(self, texts: list[str]) -> "Tokenizer":
        counts: Counter = Counter()
        for t in texts:
            counts.update(self._tokenize(t))

        # Specjalne tokeny zawsze na początku
        vocab = self.SPECIAL + sorted(
            w for w, c in counts.items() if c >= self.min_freq
        )
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        return self

    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        unk = self.stoi.get("<UNK>", 1)
        return [self.stoi.get(w, unk) for w in self._tokenize(text)]

    def decode(self, tokens: list[int], skip_special: bool = True) -> str:
        special = set(self.SPECIAL)
        words = []
        for t in tokens:
            w = self.itos.get(t, "<UNK>")
            if skip_special and w in special:
                continue
            words.append(w)
        # Sklej z inteligentną spacją (nie dodawaj spacji przed typową
        # interpunkcją).  Po modyfikacji tokenizacji apostrofy są częścią
        # słowa, więc "you'll" zostaje prawidłowo złożone bez przerw.
        out = ""
        for w in words:
            if not out:
                out = w
                continue
            # jeżeli następny token zaczyna się od litery lub apostrofu, dodaj
            # spację; w przeciwnym razie przykład (',', '.') przyczepiamy go
            # bez spacji.
            if re.match(r"[\w']", w):
                out += " " + w
            else:
                out += w
        return out

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.stoi)

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Tokenizer":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)