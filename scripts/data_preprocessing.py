# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def __getitem__(self, item):
        return self.config[item]

class MRIDataset(Dataset):

    def __init__(self, crop_dim, config, mode='train'):
        self.data_path = config['paths']['extracted_data']
        self.crop_dim = crop_dim
        self.mode = mode  # Can be 'train', 'val', or 'test'
        self.train_test_split = config['training']['batch_size'] / 10  # Example logic for splitting
        self.validate_test_split = 0.5  # Fixed split for validation
        self.number_output_classes = 2  # Assumed from dataset.json
        self.random_seed = 42  # Set fixed for reproducibility

        self.create_file_list(os.path.join(self.data_path, 'dataset.json'))
        self.prepare_split()

    def create_file_list(self, dataset_json_path):
        """
        Load file paths and dataset metadata from dataset.json.
        """
        try:
            with open(dataset_json_path, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            raise FileNotFoundError(f"File {dataset_json_path} does not exist. Ensure it is part of the project directory.")

        self.filenames = {
            idx: [
                os.path.join(self.data_path, entry['image']),
                os.path.join(self.data_path, entry['label'])
            ]
            for idx, entry in enumerate(experiment_data['training'])
        }
        self.numFiles = len(self.filenames)

    def prepare_split(self):
        """
        Split data into training, validation, and testing subsets.
        """
        indices = list(self.filenames.keys())
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        num_train = int(self.numFiles * self.train_test_split)
        num_val = int((self.numFiles - num_train) * self.validate_test_split)

        if self.mode == 'train':
            self.indices = indices[:num_train]
        elif self.mode == 'val':
            self.indices = indices[num_train:num_train + num_val]
        elif self.mode == 'test':
            self.indices = indices[num_train + num_val:]

    def z_normalize_img(self, img):
        """
        Normalize the image to have zero mean and unit variance.
        """
        img = (img - np.mean(img)) / np.std(img)
        return img

    def crop(self, img, msk, randomize):
        """
        Crop the image and mask to specified dimensions, with optional randomization.
        """
        slices = []
        is_random = randomize and np.random.rand() > 0.5

        for idx in range(len(self.crop_dim)):
            crop_len = self.crop_dim[idx]
            img_len = img.shape[idx]
            start = (img_len - crop_len) // 2
            offset = int(np.floor(start * 0.2))  # Up to 20% offset

            if offset > 0 and is_random:
                start += np.random.choice(range(-offset, offset))
                start = max(0, min(start, img_len - crop_len))

            slices.append(slice(start, start + crop_len))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk):
        """
        Apply random flips and rotations to augment data.
        """
        if np.random.rand() > 0.5:
            ax = np.random.choice(range(len(self.crop_dim)))
            img = np.flip(img, ax).copy()  # Add .copy() to avoid negative strides
            msk = np.flip(msk, ax).copy()  # Add .copy() to avoid negative strides

        if np.random.rand() > 0.5:
            rot = np.random.choice([1, 2, 3])  # 90, 180, 270 degrees
            random_axis = (0, 1)  # Assume 3D volumes are in (H, W, D) format
            img = np.rot90(img, rot, axes=random_axis).copy()  # Add .copy()
            msk = np.rot90(msk, rot, axes=random_axis).copy()  # Add .copy()

        return img, msk


    def read_nifti_file(self, idx, randomize=False):
        """
        Load and preprocess a single NIfTI file pair (image and label).
        """
        img_file, msk_file = self.filenames[idx]

        # Explicitly convert dataobj to numpy array
        img = np.asarray(nib.load(img_file).dataobj)
        msk = np.asarray(nib.load(msk_file).dataobj)

        if self.number_output_classes == 1:
            msk[msk > 0] = 1.0
        else:
            msk_temp = np.zeros((*msk.shape, self.number_output_classes))
            for channel in range(self.number_output_classes):
                msk_temp[msk == channel, channel] = 1.0
            msk = msk_temp

        img, msk = self.crop(img, msk, randomize)
        img = self.z_normalize_img(img)

        if randomize:
            img, msk = self.augment_data(img, msk)

        return img, msk


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        randomize = self.mode == 'train'
        img, msk = self.read_nifti_file(self.indices[idx], randomize=randomize)

        # Add channel dimension to img
        img = np.expand_dims(img, axis=0)  # Shape: (1, height, width, depth)

        # Convert to PyTorch tensors
        return torch.tensor(img, dtype=torch.float32), torch.tensor(msk, dtype=torch.float32)

    def visualize_sample(self, idx, save_path=None):
        """
        Visualize a single sample (image and mask).
        """
        img, msk = self.read_nifti_file(self.indices[idx], randomize=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img[:, :, img.shape[2] // 2], cmap='gray')
        axes[0].set_title("Image (Augmented)")
        
        if len(msk.shape) == 4:  # If mask is one-hot encoded
            msk = np.argmax(msk, axis=-1)  # Display the most probable class

        axes[1].imshow(msk[:, :, msk.shape[2] // 2], cmap='jet')
        axes[1].set_title("Mask (Augmented)")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D MRI Dataset Preprocessing")
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--visualize', action='store_true', help="Save visualization samples")

    args = parser.parse_args()

    config = Config(args.config)  # Load configuration

    crop_dim = (
        config['preprocessing']['tile_height'],
        config['preprocessing']['tile_width'],
        config['preprocessing']['tile_depth']
    )

    for mode in ['train', 'val', 'test']:
        dataset = MRIDataset(crop_dim, config, mode=mode)

        if args.visualize and mode == 'train':
            save_dir = os.path.join(config['paths']['extracted_data'], 'visualizations')
            os.makedirs(save_dir, exist_ok=True)
            for i in range(5):
                dataset.visualize_sample(i, save_path=os.path.join(save_dir, f'sample_{i}.png'))

        print(f"{mode.capitalize()} dataset ready with {len(dataset)} samples.")
