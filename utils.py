import re
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sentencepiece as spm


"""
Utility functions used both within the data preprocessing pipeline, and other notebooks prior to model training.
"""

# == Text Processing ==

def clean(text):
    """
    Clean and format text.
    """
    # Extract special tokens before lowercasing
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', '<SEP>', '<S0>', '<S1>', '<URL>', '<IP>', '<PATH>']

    # Entity normalisation
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text) # URLs: <URL>
    text = re.sub(r'\b\d{1,3}(\.\d{1,3}){3}\b', '<IP>', text) # IP addresses: <IP>
    text = re.sub(r'(?<!\w)(/[\w.\-]+){2,}', '<PATH>', text) # File paths: <PATH>

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    # Restore special tokens to uppercase
    for token in special_tokens:
        text = text.replace(token.lower(), token)

    return text.strip()

def filter_pair(pair, strict_question_filtering=False):
    input_text = pair['input']
    response_text = pair['response']

    # Old remove_noise logic
    if not strict_question_filtering:
        invalid_prefixes = ('!', '/', 'http')
        if any(s.strip().startswith(invalid_prefixes) for s in [input_text, response_text]):
            return False
        return True

    # Note this may filter so many samples that using dialogueText_196.csv may be more 
    # appropriate
    
    # Input must be a question
    if not input_text.strip().endswith('?'):
        return False

    # Filter social noise responses
    noise_responses = {'ok', 'okay', 'yes', 'no', 'thanks', 'lol', 'haha', 'np'}
    if response.strip().lower().rstrip('!.') in noise_responses:
        return False

    # Filter responses with mostly placeholder tokens
    placeholder_stripped = re.sub(r'<\w+>', '', response).strip()
    if len(placeholder_stripped.split()) < 2:
        return False

    # Filter multi message collapsed turns from responses
    if '<SEP>' in response:
        return False

    return True

# Setting up Byte Pair Encoding (BPE) for tokenisation
# A file will store the rules learned by the tokeniser for splitting text into subword tokens
BPE_MODEL_PATH = "./data/preprocessed/bpe.model"

sp = None
if Path(BPE_MODEL_PATH).exists():
    sp = spm.SentencePieceProcessor()
    sp.load(BPE_MODEL_PATH)

# This will be our way of accessing the learnt rules above inside the preprocessing pipeline
def tokenise(text, use_bpe=True):
    """
    Tokenising text with BPE if available, otherwise falling back to whitespace.
    """
    if use_bpe and sp is not None:
        return sp.encode(text, out_type=str)

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
    filename = 'vocab_reversed' if rev else 'vocab'
    with open(f'{PREPROCESSED_DIR}/{filename}.json', 'r') as f:
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