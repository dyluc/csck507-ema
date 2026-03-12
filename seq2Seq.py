import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, encoder_input, decoder_input, teacher_forcing_proba=0.5):
        # encoder_input: [batch_size, encoder_input_length]
        # decoder_input: [batch_size, decoder_input_length]
        
        batch_size = encoder_input.shape[0]
        decoder_input_length = decoder_input.shape[1]
        vocab_size = self.decoder.linear.out_features

        # A tensor for storing the decoder predictions (ensure tensor is on the same device as the model)
        outputs = torch.zeros(batch_size, decoder_input_length, vocab_size).to(self.device)

        if self.use_attention:
            # Forward pass through the encoder to obtain the final hidden state as well as encoder outputs at each timestep
            encoder_outputs, hidden = self.encoder(encoder_input)
        else:
            # Forward pass through the encoder to obtain the final hidden state
            hidden = self.encoder(encoder_input)

        # First decoder input is the <SOS> token for every sample
        next_decoder_input = decoder_input[:, 0]

        # Every timestep: every token in the sequence, produce a prediction
        for timestep in range(1, decoder_input_length):
            if self.use_attention:
                pred, hidden = self.decoder(next_decoder_input, hidden, encoder_outputs)
            else:
                pred, hidden = self.decoder(next_decoder_input, hidden) # Use the next decoder input and last hidden state to produce the next token prediction
            
            outputs[:, timestep, :] = pred # For all samples (tokens) in the batch, assign all decoder predictions (probability distributions) at this timestep

            # Teacher forcing (using the ground truth or model prediction with highest probability)
            # We'll apply teacher forcing with the given probability. This probability will balance learning 
            # from ground truth and learning from experience with the end goal of improving ability to generalise on unseen data (during inference)
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_proba
            next_decoder_input = decoder_input[:, timestep] if use_teacher_forcing else pred.argmax(1) # ground truth OR best prediction

        return outputs