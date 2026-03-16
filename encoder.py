import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx, hidden_dim, num_layers, dropout_proba, use_attention=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
            dropout=dropout_proba if num_layers > 1 else 0, # also apply dropout between GRU layers
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_proba)

    def forward(self, X):
        X = self.embedding(X)
        X = self.dropout(X)
        # X: [batch_size, seq_length, embed_dim]
    
        outputs, hidden = self.gru(X)
        # outputs: [batch_size, seq_length, hidden_dim * 2]
        # hidden:  [num_layers * 2, batch_size, hidden_dim]
    
        # Sum forward and backward outputs [batch_size, seq_length, hidden_dim]
        outputs = outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]
    
        # Sum forward and backward hidden states [num_layers, batch_size, hidden_dim]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim).sum(dim=1)
    
        if self.use_attention:
            return outputs, hidden
        else:
            return hidden