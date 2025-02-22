#Move dataset.json from task02_heart to extracted

import torch
import os
import shutil
import json
import numpy as np
import mlflow
import yaml
from matplotlib import pyplot as plt

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def __getitem__(self, item):
        return self.config[item]

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


def create_and_save_splits(filenames, splits_path, train_ratio=0.7, val_ratio=0.2, random_seed=42):
    np.random.seed(random_seed)
    indices = list(range(len(filenames)))
    np.random.shuffle(indices)

    num_train = int(len(indices) * train_ratio)
    num_val = int(len(indices) * val_ratio)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices
    }

    with open(splits_path, 'w') as fp:
        json.dump(splits, fp)
    print(f"Splits saved to {splits_path}")

def log_to_mlflow(config):
    """
    Logs relevant configuration parameters to MLflow.
    
    Parameters:
        config (dict): Configuration dictionary containing experiment details.
    """
    # Log general experiment settings
    mlflow.log_param("experiment_name", config['mlflow']['experiment_name'])
    mlflow.log_param("model_name", config['model']['name'])
    
    # Log training parameters
    mlflow.log_params({
        "learning_rate": config['training']['learning_rate'],
        "batch_size": config['training']['batch_size'],
        "num_epochs": config['training']['num_epochs'],
        "weight_decay": config['training'].get('weight_decay', 1e-5),
        "scheduler_step": config['training']['scheduler_step'],
        "scheduler_gamma": config['training']['scheduler_gamma'],
        "patience": config['training']['patience'],
        "val_interal": config['validation']['val_interval']
    })
    
    # Log preprocessing parameters
    mlflow.log_params({
        "crop_dim": config['preprocessing']['crop_dim'],
    })
    
    # Log paths
    mlflow.log_param("checkpoint_path", config['paths']['checkpoint'])


def save_visualizations(original_image, labels, pred_mask_tensor, batch_folder, step=10):
    """
    Save visualizations of input images, ground truth labels, and predictions.

    Args:
        original_image (numpy.ndarray): Original input volume of shape [H, W, D].
        labels (torch.Tensor): Ground truth tensor of shape [1, 1, H, W, D].
        pred_mask_tensor (torch.Tensor): Predicted segmentation tensor of shape [C, H, W, D].
        batch_folder (str): Path to the folder where visualizations should be saved.
        step (int): Step size for slices (e.g., every 10th slice).
    """
    os.makedirs(batch_folder, exist_ok=True)
    depth = original_image.shape[2]  # Get the depth (D) dimension of the volume

    for slice_idx in range(0, depth, step):
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image[:, :, slice_idx], cmap="gray")  # Use original image
        plt.title("Input Image")
        plt.subplot(1, 3, 2)
        plt.imshow(labels[0, 0, :, :, slice_idx].cpu().numpy(), cmap="jet")
        plt.title("Ground Truth")
        plt.subplot(1, 3, 3)
        plt.imshow(torch.argmax(pred_mask_tensor, dim=0).detach().cpu()[:, :, slice_idx], cmap="jet")
        plt.title("Prediction")
        save_path = os.path.join(batch_folder, f"slice_{slice_idx}.png")
        plt.savefig(save_path)
        plt.close()

