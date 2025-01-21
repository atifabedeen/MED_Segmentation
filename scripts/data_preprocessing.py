import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # For handling MRI images
import os
import numpy as np
import yaml
import json
from pathlib import Path
import torch.nn.functional as F


class HeartDataset(Dataset):
    def __init__(self, data_dir, dataset_type="train", transforms=None):
        """
        Args:
            data_dir (str): Directory containing MRI data.
            dataset_type (str): 'train', 'val', or 'test' to specify the dataset split.
            transforms (callable, optional): A function/transform to apply to the data.
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.samples = self._load_data_paths()

    def _load_data_paths(self):
        """Loads file paths for the specified dataset type."""
        dataset_file = self.data_dir / "dataset.json"
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
        
        if self.dataset_type == "train":
            if "training" not in dataset:
                raise KeyError(f"Key 'training' not found in dataset.json.")
            return [entry["image"] for entry in dataset["training"]]
        elif self.dataset_type == "test":
            if "test" not in dataset:
                raise KeyError(f"Key 'test' not found in dataset.json.")
            return dataset["test"]
        else:
            raise KeyError(f"Invalid dataset type '{self.dataset_type}'. Valid options are 'train' or 'test'.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.data_dir / self.samples[idx]
        image_data = nib.load(sample_path).get_fdata()
        # Normalize image data
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

        if self.transforms:
            image_data = self.transforms(image_data)

        return torch.tensor(image_data, dtype=torch.float32)
    
def preprocessing_pipeline(image):
    # Add preprocessing steps here
    return image  # Placeholder for transformation logic

def pad_collate_fn(batch):
    """
    Pads all tensors in a batch to the same size.
    Args:
        batch (list): List of tensors.
    Returns:
        torch.Tensor: Padded batch tensor.
    """
    max_shape = tuple(max(s[i] for s in [img.shape for img in batch]) for i in range(3))
    padded_batch = [F.pad(img, [0, max_shape[2] - img.shape[2], 
                                0, max_shape[1] - img.shape[1], 
                                0, max_shape[0] - img.shape[0]]) for img in batch]
    return torch.stack(padded_batch)


if __name__ == "__main__":
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    
    data_dir = config["paths"]["extracted_data"]
    batch_size = config["training"]["batch_size"]

    dataset = HeartDataset(data_dir=data_dir, dataset_type="train", transforms=preprocessing_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    # Example: Iterate over dataloader
    for batch in dataloader:
        print(batch.shape)  # Debugging: Print batch shape
