import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld, AsDiscreted, SpatialPadd,
    Orientationd, Spacingd, Resize
)
from monai.data import DataLoader, Dataset, CacheDataset, pad_list_data_collate
import yaml
import argparse
from scripts.utils import Config

CONFIG_FILE_PATH = "config/config.yaml"

def load_and_split_data(config, split_file="splits.json"):
    """
    Load dataset.json, split data into train, val, and test, and save to a split file.
    """
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits = json.load(f)
            train_files = splits['train']
            val_files = splits['val']
            test_files = splits['test']
    else:
        data_path = config['paths']['extracted_data']
        dataset_json_path = os.path.join(data_path, 'dataset.json')

        with open(dataset_json_path, 'r') as f:
            experiment_data = json.load(f)

        data_dicts = [
            {"image": os.path.join(data_path, entry["image"]), "label": os.path.join(data_path, entry["label"])}
            for entry in experiment_data["training"]
        ]

        test_size = config['data_split']['test_split']
        val_size = config['data_split']['val_split'] / (1 - test_size)

        train_files, test_files = train_test_split(data_dicts, test_size=test_size, random_state=42, shuffle=True)
        train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=42, shuffle=True)

        splits = {"train": train_files, "val": val_files, "test": test_files}
        with open(split_file, 'w') as f:
            json.dump(splits, f)

    return train_files, val_files, test_files


def get_transforms(config, mode='train'):
    """
    Define preprocessing and augmentation transforms using MONAI.
    """
    if mode == "train":
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest")
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config['preprocessing']['crop_dim'],
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    elif mode == "val":
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]

    elif mode == "test":
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    elif mode == "infer":
        transforms = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
        ]

    return Compose(transforms)


class DatasetManager:
    """
    Manage dataset splitting and DataLoader creation for train, val, and test modes.
    """
    def __init__(self, config, split_file="splits.json"):
        self.train_files, self.val_files, self.test_files = load_and_split_data(config, split_file=split_file)
        self.config = config

    def get_dataloader(self, mode):
        """
        Return DataLoader for the specified mode ('train', 'val', or 'test').
        """
        self.transforms = None
        if mode == 'train':
            data_files = self.train_files
            self.transforms = get_transforms(self.config, mode='train')
            shuffle = True
        elif mode == 'val':
            data_files = self.val_files
            self.transforms = get_transforms(self.config, mode='val')
            shuffle = False
        elif mode == 'test':
            data_files = self.test_files
            self.transforms = get_transforms(self.config, mode='test')
            shuffle = False
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'train', 'val', or 'test'.")

        #dataset = Dataset(data=data_files, transform=self.transforms)
        dataset = CacheDataset(data=data_files, transform=self.transforms, cache_rate=1.0)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'] if mode=="train" else self.config['validation']['batch_size'],
            shuffle=shuffle,
            num_workers=1,

        )

        return dataloader

def main():
    config = Config(CONFIG_FILE_PATH)
    dataset_manager = DatasetManager(config)

    train_loader = dataset_manager.get_dataloader('train')
    val_loader = dataset_manager.get_dataloader('val')
    test_loader = dataset_manager.get_dataloader('test')
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

if __name__ == "__main__":
    main()
