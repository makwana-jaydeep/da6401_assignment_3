

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
from torchtext.vocab import build_vocab_from_iterator


# Shared constants so every module can import them without
# hard-coding magic numbers.
SPECIALS   = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX    = 0
PAD_IDX    = 1
SOS_IDX    = 2
EOS_IDX    = 3


# ---------------------------------------------------------------------------
# spaCy model loader
# ---------------------------------------------------------------------------

def get_spacy_tokenizers():
   
    try:
        german = spacy.load("de_core_news_sm")
    except OSError:
        import subprocess
        subprocess.run(
            ["python", "-m", "spacy", "download", "de_core_news_sm"],
            check=True,
        )
        german = spacy.load("de_core_news_sm")

    try:
        english = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(
            ["python", "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        english = spacy.load("en_core_web_sm")

    return german, english


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def tokenise_german(sentence, nlp_model):
    """Run spaCy on a German string and return a list of lowercase tokens."""
    return [t.text.lower() for t in nlp_model(sentence)]


def tokenise_english(sentence, nlp_model):
    """Run spaCy on an English string and return a list of lowercase tokens."""
    return [t.text.lower() for t in nlp_model(sentence)]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Multi30kDataset(Dataset):
    

    def __init__(self, split="train", src_vocab=None, tgt_vocab=None):
        self.split = split

        # Spin up both language models once; reused across all calls.
        self.de_nlp, self.en_nlp = get_spacy_tokenizers()

        # Pull the requested split from HuggingFace.
        hf_data = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        self.raw_data = hf_data[split]

        # Build vocab only when processing training data;
        # for other splits the caller provides the training vocab.
        if src_vocab is None or tgt_vocab is None:
            assert split == "train", (
                "src_vocab and tgt_vocab must be supplied for non-training splits."
            )
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # Convert every sentence pair to integer index sequences once
        # so __getitem__ is a simple list lookup at training time.
        self.samples = self.process_data()

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def build_vocab(self):
        

        # Generator functions yield one tokenised sentence at a time to
        # avoid loading every token list into memory simultaneously.
        def de_token_stream():
            for pair in self.raw_data:
                yield tokenise_german(pair["de"], self.de_nlp)

        def en_token_stream():
            for pair in self.raw_data:
                yield tokenise_english(pair["en"], self.en_nlp)

        src_v = build_vocab_from_iterator(
            de_token_stream(),
            specials=SPECIALS,
            special_first=True,
        )
        src_v.set_default_index(UNK_IDX)

        tgt_v = build_vocab_from_iterator(
            en_token_stream(),
            specials=SPECIALS,
            special_first=True,
        )
        tgt_v.set_default_index(UNK_IDX)

        return src_v, tgt_v

    # ------------------------------------------------------------------
    # Numericalization
    # ------------------------------------------------------------------

    def process_data(self):
        
        converted = []
        for pair in self.raw_data:
            # Tokenise both sides with the respective spaCy pipeline.
            src_toks = tokenise_german(pair["de"], self.de_nlp)
            tgt_toks = tokenise_english(pair["en"], self.en_nlp)

            # Wrap with boundary markers so the model always sees
            # <sos> at position 0 and <eos> at the final position.
            src_ids = [SOS_IDX] + self.src_vocab(src_toks) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + self.tgt_vocab(tgt_toks) + [EOS_IDX]

            converted.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

        return converted

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch, pad_idx=PAD_IDX):
    
    src_seqs, tgt_seqs = zip(*batch)
    src_batch = pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_dataloaders(batch_size=128):
    # Training set -- vocab is built here.
    train_dataset = Multi30kDataset(split="train")
    shared_src_vocab = train_dataset.src_vocab
    shared_tgt_vocab = train_dataset.tgt_vocab

    # Validation and test sets reuse the training vocab.
    val_dataset  = Multi30kDataset(
        split="validation",
        src_vocab=shared_src_vocab,
        tgt_vocab=shared_tgt_vocab,
    )
    test_dataset = Multi30kDataset(
        split="test",
        src_vocab=shared_src_vocab,
        tgt_vocab=shared_tgt_vocab,
    )

    # Partial so the lambda doesn't capture a mutable default.
    pad_collate = lambda b: collate_fn(b, pad_idx=PAD_IDX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
    )
    # Single sentence at a time for cleaner greedy decode during evaluation.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pad_collate,
    )

    return train_loader, val_loader, test_loader, shared_src_vocab, shared_tgt_vocab
