#Move dataset.json from task02_heart to extracted

import torch
import os
import shutil


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

import os

def remove_hidden_files(directory):
    """
    Remove files that start with an underscore or dot from the specified directory.

    Args:
        directory (str): Path to the directory where hidden files need to be removed.

    Returns:
        int: Number of files removed.
    """
    removed_count = 0
    try:
        for filename in os.listdir(directory):
            if filename.startswith('_') or filename.startswith('.'): 
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    removed_count += 1
        return removed_count
    except Exception as e:
        print(f"Error while removing hidden files: {e}")
        return 0


def flatten_directory(root_dir):
    """
    Safely flatten a nested directory structure by moving specific nested contents
    to the specified root directory. Ensures that the function does not repeatedly
    flatten directories that are already organized.

    Args:
        root_dir (str): The root directory where all contents will be moved.
    """
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("Task"):
            for sub_item in os.listdir(folder_path):
                sub_item_path = os.path.join(folder_path, sub_item)
                dest_path = os.path.join(root_dir, sub_item)

                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(sub_item)
                    dest_path = os.path.join(root_dir, f"{base}_conflict{ext}")

                shutil.move(sub_item_path, dest_path)

            os.rmdir(folder_path)

