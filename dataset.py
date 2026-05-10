"""
dataset.py -- Multi30k Dataset Loading and Preprocessing
DA6401 Assignment 3
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
from torchtext.vocab import build_vocab_from_iterator


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def load_spacy_models():
    try:
        de_nlp = spacy.load("de_core_news_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"], check=True)
        de_nlp = spacy.load("de_core_news_sm")

    try:
        en_nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        en_nlp = spacy.load("en_core_web_sm")

    return de_nlp, en_nlp


def tokenize_de(text, nlp):
    return [tok.text.lower() for tok in nlp(text)]


def tokenize_en(text, nlp):
    return [tok.text.lower() for tok in nlp(text)]


class Multi30kDataset(Dataset):
    def __init__(self, split="train", src_vocab=None, tgt_vocab=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.de_nlp, self.en_nlp = load_spacy_models()

        raw = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        self.raw_data = raw[split]

        if src_vocab is None or tgt_vocab is None:
            assert split == "train", "Pass built vocabs for non-train splits"
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self.process_data()

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        def de_yield_tokens():
            for item in self.raw_data:
                yield tokenize_de(item["de"], self.de_nlp)

        def en_yield_tokens():
            for item in self.raw_data:
                yield tokenize_en(item["en"], self.en_nlp)

        src_vocab = build_vocab_from_iterator(
            de_yield_tokens(),
            specials=SPECIAL_TOKENS,
            special_first=True,
        )
        src_vocab.set_default_index(UNK_IDX)

        tgt_vocab = build_vocab_from_iterator(
            en_yield_tokens(),
            specials=SPECIAL_TOKENS,
            special_first=True,
        )
        tgt_vocab.set_default_index(UNK_IDX)

        return src_vocab, tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        processed = []
        for item in self.raw_data:
            src_tokens = tokenize_de(item["de"], self.de_nlp)
            tgt_tokens = tokenize_en(item["en"], self.en_nlp)

            src_indices = [SOS_IDX] + self.src_vocab(src_tokens) + [EOS_IDX]
            tgt_indices = [SOS_IDX] + self.tgt_vocab(tgt_tokens) + [EOS_IDX]

            processed.append((
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(tgt_indices, dtype=torch.long),
            ))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_idx=PAD_IDX):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def build_dataloaders(batch_size=128):
    train_ds = Multi30kDataset(split="train")
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    val_ds = Multi30kDataset(split="validation", src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_ds = Multi30kDataset(split="test", src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    pad_fn = lambda b: collate_fn(b, pad_idx=PAD_IDX)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
