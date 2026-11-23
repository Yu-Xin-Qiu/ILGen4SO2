import json
import re
from collections import Counter
from pathlib import Path

import selfies as sf


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>", "<CAT>", "<ANI>", "<PAIR>"]
SMILES_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|B|C|N|O|P|S|F|I|b|c|n|o|p|s|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


def tokenize_smiles(text):
    tokens = SMILES_PATTERN.findall(text)
    if "".join(tokens) != text:
        raise ValueError(f"SMILES tokenizer did not consume input: {text}")
    return tokens


def tokenize_selfies(text):
    return list(sf.split_selfies(text))


def build_vocab(strings, language):
    counter = Counter()
    tokenize = tokenize_smiles if language == "smiles" else tokenize_selfies
    for value in strings:
        counter.update(tokenize(value))
    ordered = SPECIAL_TOKENS + sorted(counter)
    return {token: index for index, token in enumerate(ordered)}, counter


class MolecularTokenizer:
    def __init__(self, language, vocab):
        self.language = language
        self.vocab = vocab
        self.inverse = {value: key for key, value in vocab.items()}
        self.pad_id = vocab["<PAD>"]
        self.bos_id = vocab["<BOS>"]
        self.eos_id = vocab["<EOS>"]
        self.unk_id = vocab["<UNK>"]
        self.mask_id = vocab["<MASK>"]

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["language"], payload["vocab"])

    def save(self, path):
        Path(path).write_text(
            json.dumps({"language": self.language, "vocab": self.vocab}, indent=2),
            encoding="utf-8",
        )

    def tokenize(self, text):
        return tokenize_smiles(text) if self.language == "smiles" else tokenize_selfies(text)

    def encode(self, text, ion_type=None, max_length=None):
        prefix = ["<BOS>"]
        if ion_type:
            prefix.append("<CAT>" if ion_type == "cation" else "<ANI>")
        tokens = prefix + self.tokenize(text) + ["<EOS>"]
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        if max_length is not None:
            if len(ids) > max_length:
                raise ValueError("Sequence exceeds maximum length")
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def encode_pair(self, cation, anion, max_length=None):
        tokens = ["<BOS>", "<CAT>"] + self.tokenize(cation)
        tokens += ["<PAIR>", "<ANI>"] + self.tokenize(anion) + ["<EOS>"]
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        if max_length is not None:
            if len(ids) > max_length:
                raise ValueError("Pair exceeds maximum length")
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def decode_tokens(self, ids):
        tokens = []
        for index in ids:
            token = self.inverse.get(int(index), "<UNK>")
            if token == "<EOS>":
                break
            if token not in SPECIAL_TOKENS:
                tokens.append(token)
        return "".join(tokens)

    def decode_molecule(self, ids):
        text = self.decode_tokens(ids)
        return text if self.language == "smiles" else sf.decoder(text)

    def decode_pair(self, ids):
        tokens = [self.inverse.get(int(index), "<UNK>") for index in ids]
        try:
            pair_index = tokens.index("<PAIR>")
            anion_index = tokens.index("<ANI>", pair_index + 1)
        except ValueError as exc:
            raise ValueError("Generated sequence lacks the required <PAIR>/<ANI> markers") from exc
        end_index = tokens.index("<EOS>", anion_index + 1) if "<EOS>" in tokens[anion_index + 1:] else len(tokens)
        cation_tokens = [
            token for token in tokens[2:pair_index]
            if token not in SPECIAL_TOKENS
        ]
        anion_tokens = [
            token for token in tokens[anion_index + 1:end_index]
            if token not in SPECIAL_TOKENS
        ]
        cation_repr, anion_repr = "".join(cation_tokens), "".join(anion_tokens)
        if not cation_repr or not anion_repr:
            raise ValueError("Generated pair contains an empty ion")
        if self.language == "selfies":
            return sf.decoder(cation_repr), sf.decoder(anion_repr), cation_repr, anion_repr
        return cation_repr, anion_repr, cation_repr, anion_repr
