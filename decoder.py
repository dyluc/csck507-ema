import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx, hidden_dim, num_layers, dropout_proba, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_proba if num_layers > 1 else 0
        )

        if self.use_attention:
            self.attn = nn.Linear(hidden_dim, hidden_dim)
            self.context_linear = nn.Linear(hidden_dim * 2, hidden_dim)
            
        self.linear = nn.Linear(hidden_dim, vocab_size) # Simply project from the hidden dimension size down to the vocabulary size 
        self.dropout = nn.Dropout(dropout_proba)

    def forward(self, X, hidden, encoder_outputs=None, encoder_mask=None):
        # X: [batch_size] (per sample, just one token)
        X = X.unsqueeze(1)
        # X: [batch_size, 1] (add seq_length dimension, which is just 1, since we're only processing one token at a time)
        X = self.embedding(X)
        X = self.dropout(X)
        # X: [batch_size, 1, embed_dim]
        output, hidden = self.gru(X, hidden)
        # output: [batch_size, 1, hidden_dim] (?)
        # hidden: [num_layers, batch_size, hidden_dim] (the next hidden state to use at the next timestep)

        if self.use_attention:
            scale = encoder_outputs.size(-1) ** 0.5
            # Compute attention scores between the decoder state and every encoder timestep
            attn_scores = torch.bmm(
                output, 
                self.attn(encoder_outputs).transpose(1, 2)
            ) / scale
            # ignore padded encoder positions.
            if encoder_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    encoder_mask.unsqueeze(1) == 0,
                    -1e9
                )
            # Convert raw attention scores into probabilities
            attn_weights = torch.softmax(attn_scores, dim=-1) # Softmax ensures the weights sum to 1 across the encoder sequence
            # Compute the context vector
            context = torch.bmm(attn_weights, encoder_outputs) # This is a weighted sum of encoder hidden states using the attention weights
            # Combine decoder state and context vector
            combined = torch.cat((output, context), dim=2)
            # Project the combined vector back to hidden size
            combined = self.context_linear(combined) # mixes decoder information and context information

            pred = self.linear(combined.squeeze(1)) # Convert back to [batch_size] and project to vocabulary
            # pred: [batch_size, hidden_size] (this is our next token prediction!)
        else:
            pred = self.linear(output.squeeze(1)) # Convert back to [batch_size] and project to vocabulary
            # pred: [batch_size, vocab_size] (this is our next token prediction!)

        return pred, hidden