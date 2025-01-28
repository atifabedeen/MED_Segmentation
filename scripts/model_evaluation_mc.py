import os
import torch
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    SaveImaged,
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from scripts.data_preprocessing import DatasetManager, Config
from scripts.model_loader import load_model_from_config
from scripts.utils import load_checkpoint, save_visualizations
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score
from monai.handlers.utils import from_engine

def visualize_uncertainty_map(image, mc_mean, mc_variance, slice_idx, save_path=None):
    """
    Visualize the original image, mean prediction (foreground channel), and uncertainty (variance, foreground channel).
    Args:
        image (torch.Tensor): Original image with shape (H, W, D).
        mc_mean (torch.Tensor): Mean prediction from MC Dropout with shape (C, H, W, D).
        mc_variance (torch.Tensor): Uncertainty map (variance) with shape (C, H, W, D).
        slice_idx (int): Index for slicing 3D image.
        save_path (str): Path to save the visualization.
    """
    # Select the foreground channel (index 1)
    mc_mean_foreground = mc_mean[1, :, :, :].cpu().numpy()
    mc_variance_foreground = mc_variance[1, :, :, :].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image[:, :, slice_idx], cmap="gray")
    axes[0].set_title("Original Image")

    axes[1].imshow(mc_mean_foreground[:, :, slice_idx], cmap="jet", alpha=0.7)
    axes[1].set_title("Mean Prediction (Foreground)")

    axes[2].imshow(mc_variance_foreground[:, :, slice_idx], cmap="viridis", alpha=0.7)
    axes[2].set_title("Uncertainty (Variance, Foreground)")

    if save_path:
        plt.savefig(save_path)
    plt.show()



def enable_dropout(model):
    """Enable MC Dropout by setting all dropout layers to train mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout3d):
            module.train()

def inference_mc_dropout(config, model, test_loader, transforms, device, mc_samples=10):
    """Runs inference with MC Dropout, computes metrics, and logs results."""
    os.makedirs(config['paths']['results'], exist_ok=True)
    log_file = os.path.join(config['paths']['results'], "metrics_log_mc.txt")

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            AsDiscreted(keys="label", to_onehot=2),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="mc_seg", resample=False),
        ]
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(percentile=95, include_background=False)

    roi_size = config['validation'].get('roi_size', (96, 96, 96))
    sw_batch_size = config['validation'].get('sw_batch_size', 4)
    pred_loader = LoadImage()

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            original_image = images[0, 0].cpu().numpy()

            enable_dropout(model)
            mc_preds = torch.stack(
                [sliding_window_inference(images, roi_size, sw_batch_size, model) for _ in range(mc_samples)]
            )
            mc_mean = torch.mean(mc_preds, dim=0)
            mc_variance = mc_preds.var(dim=0)

            batch_data["pred"] = mc_mean
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

            pred_mask, gt_mask = from_engine(["pred", "label"])(batch_data)
            gt_mask_tensor = torch.cat(gt_mask, dim=0)
            pred_mask_tensor = torch.cat(pred_mask, dim=0)
            original_image = pred_loader(pred_mask[0].meta["filename_or_obj"])
            print(original_image.shape)
            print(mc_mean.shape)
            print(mc_variance.shape)
            print(images.shape)
            dice_score = dice_metric(y_pred=pred_mask, y=gt_mask).item()
            hausdorff_distance = hausdorff_metric(y_pred=pred_mask, y=gt_mask).item()
            slice_idx = mc_mean.shape[-1] // 2  
            print(slice_idx)
            save_path = os.path.join(config['paths']['results'], f"uncertainty_map_{idx}.png")
            visualize_uncertainty_map(images[0, 0].cpu().numpy(), mc_mean[0], mc_variance[0], slice_idx, save_path)

            metrics = {
                "Image Index": idx,
                "Dice Score": dice_score,
                "Hausdorff Distance": hausdorff_distance,
            }

            with open(log_file, "a") as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

            batch_folder = os.path.join(config['paths']['results'], f"visualizations/batch_{idx}")
            os.makedirs(batch_folder, exist_ok=True)

            #save_visualizations(original_image, labels, pred_mask_tensor, batch_folder, step=10)





if __name__ == "__main__":
    config = Config("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_config('config/config.yaml').to(device)
    model = torch.nn.DataParallel(model)
    load_checkpoint(os.path.join("checkpoints", f"best_model_{config['model']['name'].lower()}.pth"), model)

    model.eval()

    dataset_manager = DatasetManager(config)
    test_loader = dataset_manager.get_dataloader("test")

    inference_mc_dropout(config, model, test_loader, dataset_manager.transforms, device)
