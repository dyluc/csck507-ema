import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import time
from bleu import compute_bleu

def get_teacher_forcing_proba(epoch, total_epochs, start_proba, end_proba, shift_epochs=0):
    """
    Start with a simple linear decay from start ratio to end ratio over the total epochs. Shifting can 
    be used to adjust the epoch at which the end_proba is reached. Can be negative or positive.
    """
    decay = start_proba - (start_proba - end_proba) * (epoch / (total_epochs - shift_epochs - 1))
    return max(decay, end_proba)

def _train_epoch(model, loader, encoder_optimiser, decoder_optimiser, criterion, device, clip_max_norm, teacher_forcing_proba, padding_idx):
    model.train() # Enables training mode (includes enabling dropout)
    total_loss = 0
    total_batches = len(loader)
    log_interval = max(1, int(total_batches * 0.05)) # Every 5%

    for batch_idx, (encoder_input_batch, decoder_input_batch, decoder_target_batch) in enumerate(loader):
        # Remember tensors and models must be on the same device
        encoder_input_batch = encoder_input_batch.to(device)
        decoder_input_batch = decoder_input_batch.to(device)
        decoder_target_batch = decoder_target_batch.to(device)

        encoder_optimiser.zero_grad()
        decoder_optimiser.zero_grad()
        
        if model.use_attention: # our attention model takes the padding token id
            predictions = model(encoder_input_batch, decoder_input_batch, 
                              teacher_forcing_proba=teacher_forcing_proba,
                              padding_idx=padding_idx)
        else:
            predictions = model(encoder_input_batch, decoder_input_batch, 
                              teacher_forcing_proba=teacher_forcing_proba)
        
        predictions = predictions.permute(0, 2, 1) # Need to reshape for CrossEntropyLoss, this just reorders dimensions
        loss = criterion(predictions, decoder_target_batch) # Compute loss
        loss.backward() # BPTT
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm) # Gradient clipping comes before weight updates
        encoder_optimiser.step()
        decoder_optimiser.step()
        total_loss += loss.item() 

        # Log every 25% of batches (or whatever log_percent you set)
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress = 100 * (batch_idx + 1) / total_batches
            print(f'  [{progress:.0f}%] Batch {batch_idx+1}/{total_batches} | Avg Loss: {avg_loss:.4f}')

    return total_loss / len(loader) # Average loss per batch

def _val_epoch(model, loader, criterion, device, padding_idx):
    model.eval() # Deactivates dropout
    total_loss = 0
    with torch.no_grad(): # Disable gradient computation and tracking
        for encoder_input_batch, decoder_input_batch, decoder_target_batch in loader:
            encoder_input_batch = encoder_input_batch.to(device)
            decoder_input_batch = decoder_input_batch.to(device)
            decoder_target_batch = decoder_target_batch.to(device)

            # no teacher forcing during evaluation
            if model.use_attention:
                predictions = model(encoder_input_batch, decoder_input_batch, 
                                  teacher_forcing_proba=0.0,
                                  padding_idx=padding_idx)
            else:
                predictions = model(encoder_input_batch, decoder_input_batch, 
                                  teacher_forcing_proba=0.0)
            
            predictions = predictions.permute(0, 2, 1)
            loss = criterion(predictions, decoder_target_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, vocab_reversed, config, device, checkpoint_dir, hyperparams):

    # Training parameters
    EPOCHS = hyperparams['EPOCHS']
    CLIP_MAX_NORM = hyperparams['CLIP_MAX_NORM']
    TEACHER_FORCING_PROBA = hyperparams['TEACHER_FORCING_PROBA'] # Now the starting proba
    ENCODER_LEARNING_RATE = hyperparams['ENCODER_LEARNING_RATE']
    DECODER_LEARNING_RATE = hyperparams['DECODER_LEARNING_RATE']
    
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / f'{checkpoint_dir.name}_baseline.pt'

    criterion = nn.CrossEntropyLoss(ignore_index=config['PAD_IDX'])
    encoder_optimiser = Adam(model.parameters(), lr=ENCODER_LEARNING_RATE)
    decoder_optimiser = Adam(model.parameters(), lr=DECODER_LEARNING_RATE)

    existing_checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1])
    )

    if model_path.exists():
        print('Loading model...')
        model.load_state_dict(torch.load(model_path, weights_only=False).state_dict())
        return
    elif existing_checkpoints:
        print('Loading checkpoints...')
        latest = existing_checkpoints[-1]
        checkpoint = torch.load(latest, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        encoder_optimiser.load_state_dict(checkpoint['encoder_optimiser_state_dict'])
        decoder_optimiser.load_state_dict(checkpoint['decoder_optimiser_state_dict'])
        start_epoch = int(latest.stem.split('_')[-1])
        print(f'Resuming from epoch {start_epoch}/{EPOCHS}')
    else:
        print('No checkpoints, training model from scratch...')
        start_epoch = 0

    # Main training loop
    epoch_times = []
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()

        current_epoch_tf_proba = get_teacher_forcing_proba(
            epoch=epoch,
            total_epochs=EPOCHS,
            start_proba=TEACHER_FORCING_PROBA,
            end_proba=0.0
        )
        
        train_loss = _train_epoch(
            model, 
            train_loader, 
            encoder_optimiser,
            decoder_optimiser,
            criterion,
            device,
            CLIP_MAX_NORM,
            current_epoch_tf_proba,
            config['PAD_IDX']
        )
        val_loss = _val_epoch(
            model, 
            val_loader, 
            criterion, 
            device,
            config['PAD_IDX']
        )
        bleu = compute_bleu(
            model, 
            val_loader,
            vocab_reversed, 
            config, 
            device
        )
        elapsed = time.time() - start
        epoch_times.append(elapsed)
        
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {bleu:.4f} | Time: {elapsed:.2f}s')
    
        # Checkpoint the state dict (smaller file size, doesn't preserve architecture)
        torch.save({
            'model_state_dict': model.state_dict(),
            'encoder_optimiser_state_dict': encoder_optimiser.state_dict(),
            'decoder_optimiser_state_dict': decoder_optimiser.state_dict(),
        }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save full model
    torch.save(model, model_path)
    
    # Print benchmarks
    if epoch_times:
        print(f'\n--- Benchmark Summary ---')
        print(f'Device: {device}')
        print(f'Total training time: {sum(epoch_times):.2f}s')
        print(f'Average epoch time: {sum(epoch_times)/len(epoch_times):.2f}s')
        print(f'Fastest epoch: {min(epoch_times):.2f}s')
        print(f'Slowest epoch: {max(epoch_times):.2f}s')
    else:
        print('Training already complete!')
    