# Placeholder
#Move dataset.json from task02_heart to extracted

import torch
import os

def save_checkpoint(model, optimizer, filepath, epoch=None, val_loss=None):
    """Saves model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        filepath (str): Path to save the checkpoint.
        epoch (int, optional): Current epoch number. Default is None.
        val_loss (float, optional): Validation loss. Default is None.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Loads model and optionally optimizer state from a checkpoint file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into. Default is None.

    Returns:
        dict: Additional information stored in the checkpoint (e.g., epoch, val_loss).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} does not exist.")

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {filepath}")

    return {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict']}
