import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx, hidden_dim, num_layers, dropout_proba, use_attention=False):
        super().__init__()
        
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=padding_idx # Pad out to sequence length
        )
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim, # The dimension of the hidden state vector maintained through timesteps
            num_layers=num_layers,
            batch_first=True, # This just moves batch to the first dimension, to match our dataloader implementation
            dropout=dropout_proba if num_layers > 1 else 0 # also apply dropout between GRU layers
        )
        self.dropout = nn.Dropout(dropout_proba)

    def forward(self, X):
        # X: [batch_size, seq_length]
        X = self.embedding(X)
        X = self.dropout(X) # Applying dropout here is standard for seq2seq, it's the largest parameter space in the network
        # X: [batch_size, seq_length, embed_dim]
        outputs, hidden = self.gru(X)
        # outputs: [batch_size, seq_length, hidden_dim] (
        # hidden: [num_layers, batch_size, hidden_dim] (our final hidden state)

        if self.use_attention:
            # return outputs at each timestep as well as final hidden state
            return outputs, hidden
        else:
            # We'll ignore the outputs in case not using attention
            return hidden