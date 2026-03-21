import torch
import argparse
from utils import encode_sequence, clean, load_vocab, load_config

SKIP_TOKENS = {'<SEP>', '<S0>', '<S1>'}

def generate_response(model, text, vocab, vocab_reversed, config, device, max_len=30, temperature=1.0, top_k=5, repetition_penalty=1.0):
    model.eval()
    encoder_input = torch.tensor(
        encode_sequence(clean(text), vocab, config['MAX_LENGTH']),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(encoder_input)
        if not model.use_attention and hidden.size(0) == 1:
            hidden = hidden.repeat(2, 1, 1)
        next_token = torch.tensor([config['SOS_IDX']], dtype=torch.long).to(device)
        tokens = []
        
        encoder_mask = (encoder_input != config['PAD_IDX']) if model.use_attention else None

        for _ in range(max_len):
            if model.use_attention:
                pred, hidden = model.decoder(next_token, hidden, encoder_outputs, encoder_mask)
            else:
                pred, hidden = model.decoder(next_token, hidden)
            scaled = pred / temperature

            # This is an inference time penalty we can apply to the model
            for token_id in set([vocab[t] for t in tokens if t in vocab]):
                scaled[0][token_id] -= repetition_penalty

            top_k_values, top_k_indices = torch.topk(scaled, top_k, dim=-1)
            probs = torch.softmax(top_k_values, dim=-1)
            next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1)).squeeze(-1)

            if next_token.item() == config['EOS_IDX']:
                break

            token_str = vocab_reversed[str(next_token.item())]
            if token_str not in SKIP_TOKENS:
                tokens.append(token_str)

    return ' '.join(tokens)


def main():
    parser = argparse.ArgumentParser(description='Inference script for seq2seq model')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model file')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=1, help='Top-k sampling')
    parser.add_argument('--max_len', type=int, default=30, help='Max response length')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading model from {args.model}...')
    model = torch.load(args.model, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    vocab = load_vocab()
    vocab_reversed = load_vocab(rev=True)
    config = load_config()

    print('Type message or "exit" to exit\n')

    while True:
        text = input('Input: ').strip()
        if text.lower() == 'exit':
            break
        if not text:
            continue

        # We have to prepend a speaker token if not present, this is the first token the model has
        # seen throughout its entire training
        if not text.startswith('<S0>'):
            text = f'<S0> {text}'

        response = generate_response(
            model, text, vocab, vocab_reversed, config, device,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f'Response: {response}')

if __name__ == '__main__':
    main()