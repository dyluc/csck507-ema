import re
import json
import os
from pathlib import Path
from collections import Counter

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


"""
Utility functions for preprocessing + model training
Supports BOTH:
1) Word-level tokenizer (legacy)
2) BPE tokenizer (recommended)
"""

# ==============================================================================
# == Constants ==
# ==============================================================================

PREPROCESSED_DIR = './data/preprocessed'
BPE_DIR = './data/bpe'

# Keep all conversation structure + entity tokens as specials
BASE_SPECIAL_TOKENS = ['<PAD>', '<SOS>',
                       '<EOS>', '<UNK>', '<SEP>', '<S0>', '<S1>']
ENTITY_TOKENS = ['<url>', '<path>', '<ip>',
                 '<user>', '<version>', '<email>', '<cmd>']
SPECIAL_TOKENS = BASE_SPECIAL_TOKENS + ENTITY_TOKENS

# ==============================================================================
# == Text Processing ==
# ==============================================================================

_ENTITY_PATTERNS = [
    # 1) URLs first
    (re.compile(r'https?://\S+|ftp://\S+|www\.\S+', re.IGNORECASE), '<url>'),

    # 2) Email
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.IGNORECASE), '<email>'),

    # 3) IPv4
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '<ip>'),

    # 4) Windows path
    (re.compile(r'[A-Za-z]:\\(?:[\w\s.\-]+\\)*[\w\s.\-]*'), '<path>'),

    # 5) Unix path (2+ segments)
    (re.compile(r'(?:~|\.{1,2})?/[\w.\-@+]+(?:/[\w.\-@+]*)+'), '<path>'),

    # 6) Version
    (re.compile(r'\bv?\d+\.\d+(?:\.\d+)*\b', re.IGNORECASE), '<version>'),

    # 7) IRC username style <john_doe>, avoid known tokens
    (re.compile(
        r'<(?!(?:pad|sos|eos|unk|sep|s0|s1|url|path|ip|user|version|email|cmd)>)'
        r'[a-zA-Z_][a-zA-Z0-9_\-]{1,30}>'
    ), '<user>'),

    # 8) Common commands
    (re.compile(r'\b(?:apt-get|apt|dpkg|sudo|chmod|chown|grep|awk|sed|wget|curl|pip3?|python3?)\b', re.IGNORECASE), '<cmd>'),
]


def clean(text: str) -> str:
    """
    Normalize noisy entities while preserving conversation structure tokens.
    """
    text = str(text)

    # Entity normalization first
    for pattern, token in _ENTITY_PATTERNS:
        text = pattern.sub(token, text)

    # Lowercase
    text = text.lower()

    # Remove residual tags BUT keep all special tokens
    # preserved tokens: <pad><sos><eos><unk><sep><s0><s1><url><path><ip><user><version><email><cmd>
    text = re.sub(
        r'<(?!/?(?:pad|sos|eos|unk|sep|s0|s1|url|path|ip|user|version|email|cmd)>)[^>]+>',
        '',
        text
    )

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_noise(text: str) -> bool:
    """
    Filter obvious non-conversational lines.
    """
    text = str(text).strip()
    if len(text) < 2:
        return False
    if text.startswith('!'):
        return False
    if text.startswith('/'):
        return False
    # raw url line
    if text.startswith('http'):
        return False
    return True


def tokenise(text, bpe=None):
    if bpe is not None:
        return bpe.tokenise(text)
    return str(text).split()


def encode_sequence(text, vocab, max_len, bpe=None):
    if bpe is not None:
        return bpe.encode_sequence(text, max_len)

    tokens = tokenise(text)[:max_len]
    vals = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    return vals + [vocab['<PAD>']] * (max_len - len(vals))


# ==============================================================================
# == Word-level vocab helpers ==
# ==============================================================================

def build_word_vocab(train_df: pd.DataFrame, min_token_freq: int = 5):
    """
    Build legacy word-level vocab from train split only.
    """
    counter = Counter()
    for text in train_df['input']:
        counter.update(tokenise(text))
    for text in train_df['response']:
        counter.update(tokenise(text))

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for word, count in counter.items():
        if count >= min_token_freq and word not in vocab:
            vocab[word] = len(vocab)

    vocab_reversed = {idx: word for word, idx in vocab.items()}
    return vocab, vocab_reversed, counter


# ==============================================================================
# == BPE ==
# ==============================================================================

def train_bpe(train_csv_path, output_dir=BPE_DIR, vocab_size=16000, min_frequency=2):
    """
    Train BPE tokenizer from train.csv (input + response columns).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv_path)
    if 'input' not in df.columns or 'response' not in df.columns:
        raise ValueError(
            "train.csv must contain 'input' and 'response' columns")

    corpus_path = os.path.join(output_dir, '_corpus_tmp.txt')
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(str(row['input']) + '\n')
            f.write(str(row['response']) + '\n')

    tokenizer = Tokenizer(BPE(unk_token='<UNK>'))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    os.remove(corpus_path)

    tokenizer_path = os.path.join(output_dir, 'bpe_tokenizer.json')
    tokenizer.save(tokenizer_path)

    vocab = tokenizer.get_vocab()
    with open(os.path.join(output_dir, 'vocab_bpe.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'vocab_bpe_reversed.json'), 'w', encoding='utf-8') as f:
        json.dump({str(v): k for k, v in vocab.items()},
                  f, ensure_ascii=False, indent=2)

    return load_bpe(output_dir)


def load_bpe(bpe_dir=BPE_DIR):
    return BPETokenizer(os.path.join(bpe_dir, 'bpe_tokenizer.json'))


class BPETokenizer:
    def __init__(self, tokenizer_path):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"BPE tokenizer not found: {tokenizer_path}\n"
                f"Run train_bpe('./data/preprocessed/train.csv') first."
            )
        self._tok = Tokenizer.from_file(tokenizer_path)
        self.vocab = self._tok.get_vocab()

        self.pad_id = self.vocab['<PAD>']
        self.sos_id = self.vocab['<SOS>']
        self.eos_id = self.vocab['<EOS>']
        self.unk_id = self.vocab['<UNK>']
        self.sep_id = self.vocab['<SEP>']
        self.s0_id = self.vocab['<S0>']
        self.s1_id = self.vocab['<S1>']

    def get_vocab_size(self):
        return self._tok.get_vocab_size()

    def tokenise(self, text):
        return self._tok.encode(str(text), add_special_tokens=False).tokens

    def encode_sequence(self, text, max_len):
        ids = self._tok.encode(
            str(text), add_special_tokens=False).ids[:max_len]
        return ids + [self.pad_id] * (max_len - len(ids))

    def encode_with_sos(self, text, max_len):
        ids = self._tok.encode(
            str(text), add_special_tokens=False).ids[:max_len - 1]
        ids = [self.sos_id] + ids
        return ids + [self.pad_id] * (max_len - len(ids))

    def encode_with_eos(self, text, max_len):
        ids = self._tok.encode(
            str(text), add_special_tokens=False).ids[:max_len - 1]
        ids = ids + [self.eos_id]
        return ids + [self.pad_id] * (max_len - len(ids))

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)


# ==============================================================================
# == Dataset / Dataloader ==
# ==============================================================================

class DialogueDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, vocab, max_len: int, bpe=None):
        self.pairs = pairs.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.bpe = bpe

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text = self.pairs.loc[idx, 'input']
        response_text = self.pairs.loc[idx, 'response']

        if self.bpe is not None:
            encoder_input = self.bpe.encode_sequence(input_text, self.max_len)
            decoder_input = self.bpe.encode_with_sos(
                response_text, self.max_len)
            decoder_target = self.bpe.encode_with_eos(
                response_text, self.max_len)
        else:
            encoder_input = encode_sequence(
                input_text, self.vocab, self.max_len)

            response_tokens = tokenise(response_text)[:self.max_len - 1]
            encoded_response = [self.vocab.get(
                t, self.vocab['<UNK>']) for t in response_tokens]

            decoder_input = [self.vocab['<SOS>']] + encoded_response
            decoder_target = encoded_response + [self.vocab['<EOS>']]

            remaining = self.max_len - len(decoder_input)
            decoder_input = decoder_input + [self.vocab['<PAD>']] * remaining
            decoder_target = decoder_target + [self.vocab['<PAD>']] * remaining

        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_target, dtype=torch.long)
        )


def get_dataloaders(train_set, val_set, test_set, vocab, max_length, batch_size, bpe=None, num_workers=0):
    train_dataset = DialogueDataset(train_set, vocab, max_length, bpe)
    val_dataset = DialogueDataset(val_set, vocab, max_length, bpe)
    test_dataset = DialogueDataset(test_set, vocab, max_length, bpe)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ==============================================================================
# == IO helpers ==
# ==============================================================================

def load_sets(preprocessed_dir=PREPROCESSED_DIR):
    train_set = pd.read_csv(f'{preprocessed_dir}/train.csv')
    val_set = pd.read_csv(f'{preprocessed_dir}/val.csv')
    test_set = pd.read_csv(f'{preprocessed_dir}/test.csv')
    return train_set, val_set, test_set


def load_vocab(rev=False, preprocessed_dir=PREPROCESSED_DIR, name='vocab'):
    filename = f'{name}_reversed' if rev else name
    with open(f'{preprocessed_dir}/{filename}.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_config(preprocessed_dir=PREPROCESSED_DIR):
    with open(f'{preprocessed_dir}/config.json', 'r', encoding='utf-8') as f:
        return json.load(f)
