from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch

def compute_bleu(model, loader, vocab_reversed, config, device, n_batches=10):
    model.eval()
    references, hypotheses = [], []
    smoother = SmoothingFunction().method1
    print('Computing BLEU for epoch...')
    
    with torch.no_grad():
        for i, (enc_batch, dec_in_batch, dec_tgt_batch) in enumerate(loader):
            if i >= n_batches:
                break
            if i % 3 == 0:
                print(f'  BLEU: batch {i+1}/{n_batches}...')
            enc_batch = enc_batch.to(device)
            
            for j in range(enc_batch.shape[0]):
                ref = [vocab_reversed.get(str(t.item()), '<UNK>') 
                       for t in dec_tgt_batch[j] 
                       if t.item() not in (config['PAD_IDX'], config['EOS_IDX'], config['SOS_IDX'])]
                
                encoder_input = enc_batch[j].unsqueeze(0)

                # need to handle both model encoders (with attention and without) -> the shapes are different
                encoder_result = model.encoder(encoder_input)
                if isinstance(encoder_result, tuple):
                    encoder_outputs, hidden = encoder_result
                else:
                    encoder_outputs = None
                    hidden = encoder_result

                next_token = torch.tensor([config['SOS_IDX']], dtype=torch.long).to(device)
                hyp = []
                for _ in range(config['MAX_LENGTH']):
                    if encoder_outputs is not None:
                        pred, hidden = model.decoder(next_token, hidden, encoder_outputs)
                    else:
                        pred, hidden = model.decoder(next_token, hidden)
                    next_token = pred.argmax(1)
                    if next_token.item() == config['EOS_IDX']:
                        break
                    hyp.append(vocab_reversed.get(str(next_token.item()), '<UNK>'))
                
                if ref and hyp:
                    references.append([ref])
                    hypotheses.append(hyp)
    
    return corpus_bleu(references, hypotheses, smoothing_function=smoother)