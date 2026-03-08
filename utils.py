import re
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader

"""
Utility functions used both within the data preprocessing pipeline, and other notebooks prior to model experimentation.
"""

# == Text Processing ==

def clean(text):
    """
    Clean and format text.
    """
    text = text.lower() # lowercase (normalise vocabulary size)
    text = re.sub(r'http\S+', '', text) # remove URLs
    # text = re.sub(r'[^a-z0-9\s?.!,]', '', text) # retain specific punctuation (perhaps a bit too destructive)
    text = re.sub(r'\s+', ' ', text) # format whitespace
    return text.strip()

def tokenise(text):
    """
    Split on whitespace.
    """
    return text.split()

def encode_sequence(text, vocab, max_len):
    """
    Encode text into id sequences using the provided vocab dictionary. Truncate and pad.
    """
    tokens = tokenise(text)[:max_len] # Truncate to maximum sequence length
    vals = [vocab.get(t, vocab['<UNK>']) for t in tokens] # Get token, or default to <UNK>
    remaining = max_len - len(vals)
    vals = vals + [vocab['<PAD>']] * remaining # Pad to the max length
    
    return vals


# == Dataset ==

class DialogueDataset(Dataset):
    def __init__(self, pairs, vocab, max_len):
        self.pairs = pairs.reset_index(drop=True) # pairs is a DataFrame
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text = self.pairs.loc[idx, 'input']
        response_text = self.pairs.loc[idx, 'response']

        # encoder_input: Encoded input sequence (for the encoder)
        encoder_input = encode_sequence(input_text, self.vocab, self.max_len)

        # Encoded response (used to construct decoder input and target)
        response_tokens = tokenise(response_text)[:self.max_len - 1] # Chop off a token to replace with either SOS or EOS
        encoded_response = [self.vocab.get(t, self.vocab['<UNK>']) for t in response_tokens]

        # decoder_input: Encoded response with <SOS> prepended (pad to the max length)
        decoder_input = [self.vocab['<SOS>']] + encoded_response
        remaining = self.max_len - len(decoder_input)
        decoder_input = decoder_input + [self.vocab['<PAD>']] * remaining

        # decoder_target: Encoded response with <EOS> appended (pad to the max length)
        decoder_target = encoded_response + [self.vocab['<EOS>']]
        decoder_target = decoder_target + [self.vocab['<PAD>']] * remaining

        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_target, dtype=torch.long)
        )
    

# == Loaders ==

PREPROCESSED_DIR = './data/preprocessed'

def load_sets():
    train_set = pd.read_csv(f'{PREPROCESSED_DIR}/train.csv')
    val_set = pd.read_csv(f'{PREPROCESSED_DIR}/val.csv')
    test_set = pd.read_csv(f'{PREPROCESSED_DIR}/test.csv')

    return train_set, val_set, test_set

def load_vocab(rev=False):
    with open(f'{PREPROCESSED_DIR}/{'vocab_reversed' if rev else 'vocab'}.json', 'r') as f:
        return json.load(f)

def load_config():
    with open(f'{PREPROCESSED_DIR}/config.json', 'r') as f:
        return json.load(f)

def get_dataloaders(train_set, val_set, test_set, vocab, max_length, batch_size):
    
    # Lets create our PyTorch Datasets and DataLoaders!
    train_dataset = DialogueDataset(train_set, vocab, max_length)
    val_dataset = DialogueDataset(val_set, vocab, max_length)
    test_dataset = DialogueDataset(test_set, vocab, max_length)

    # Shuffle to ensure all batches get a good representation of samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader