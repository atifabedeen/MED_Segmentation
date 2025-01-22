import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose, Activations, AsDiscrete
from data_preprocessing import MRIDataset
from model_loader import load_model_from_config
from utils import load_checkpoint
import yaml
from matplotlib import pyplot as plt

def visualize_segmentation(image, pred, save_path=None):
    """Visualizes and optionally saves the overlay of the image and prediction.

    Args:
        image (numpy.ndarray): 3D image volume.
        pred (numpy.ndarray): Predicted mask.
        save_path (str, optional): Path to save the visualization. Defaults to None.
    """
    slice_idx = image.shape[2] // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image[:, :, slice_idx], cmap="gray")
    axes[0].set_title("Image")

    axes[1].imshow(pred[:, :, slice_idx], cmap="jet", alpha=0.5)
    axes[1].set_title("Prediction")

    if save_path:
        plt.savefig(save_path)
    plt.show()

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model_from_config('config/config.yaml').to(device)
load_checkpoint(config['paths']['checkpoint'], model)
model.eval()

post_transforms = Compose([
    Activations(sigmoid=True),  
    AsDiscrete(threshold=0.5)
])

# Load test data
crop_dim = (
    config['preprocessing']['tile_height'],
    config['preprocessing']['tile_width'],
    config['preprocessing']['tile_depth']
)
data_test = MRIDataset(crop_dim, config, mode='test')
test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

def run_inference(model, test_loader, post_transforms):
    os.makedirs(config['paths']['results'], exist_ok=True)
    for data in test_loader:  
        data = data.to(device)

        with torch.no_grad():
            outputs = model(data)
            preds = post_transforms(outputs)

        pred = preds.cpu().numpy()[0, 0]
        img = data.cpu().numpy()[0, 0]

        save_path = os.path.join(config['paths']['results'], f"prediction_{np.random.randint(1e6)}.png")
        visualize_segmentation(img, pred, save_path)


if __name__ == "__main__":
    run_inference(model, test_loader, post_transforms)
