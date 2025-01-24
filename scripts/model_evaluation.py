import os
import torch
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    Spacing,
    LoadImage,
    SaveImaged,

)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from data_preprocessing import DatasetManager, Config, get_transforms
from model_loader import load_model_from_config
from utils import load_checkpoint
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score
from monai.handlers.utils import from_engine

def visualize_segmentation(image, pred, save_path=None):
    """Visualizes and optionally saves the overlay of the image and prediction."""
    # Calculate the middle slice index for both the image and the prediction
    image_slice_idx = image.shape[2] // 2
    pred_slice_idx = pred.shape[2] // 2

    # If pred has multiple channels (e.g., one-hot), take one channel or merge them
    if pred.shape[0] > 1:
        pred = pred.argmax(axis=0)  # Take the class with the highest probability (e.g., for softmax output)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the middle slice of the image
    axes[0].imshow(image[:, :, image_slice_idx], cmap="gray")
    axes[0].set_title("Image")

    # Display the middle slice of the prediction
    axes[1].imshow(pred[:, :, pred_slice_idx], cmap="jet", alpha=0.5)
    axes[1].set_title("Prediction")

    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_additional_metrics(preds, labels):
    """Computes IoU, Precision, Recall, and Specificity."""
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    iou = jaccard_score(labels_flat, preds_flat, average='binary')
    precision = precision_score(labels_flat, preds_flat)
    recall = recall_score(labels_flat, preds_flat)
    specificity = recall_score(1 - labels_flat, 1 - preds_flat)

    return iou, precision, recall, specificity


def log_metrics(metrics, log_file):
    """Logs metrics to a specified file."""
    with open(log_file, "a") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")


def run_inference(config, model, test_loader, transforms, device):
    """Runs inference, computes metrics, and logs results."""
    os.makedirs(config['paths']['results'], exist_ok=True)
    log_file = os.path.join(config['paths']['results'], "metrics_log.txt")

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
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(percentile=95, include_background=False)
    spacing_transform = Spacing(pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"))

    roi_size = config['validation'].get('roi_size', (160, 160, 160))
    sw_batch_size = config['validation'].get('sw_batch_size', 4)
    pred_loader = LoadImage()
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            original_image = images[0, 0].cpu().numpy()  # Extract the 3D image volume

            # Perform sliding window inference
            batch_data["pred"] = sliding_window_inference(images, roi_size, sw_batch_size, model)
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

            # Extract the prediction mask
            pred_mask, gt_mask = from_engine(["pred", "label"])(batch_data)
            original_image = pred_loader(pred_mask[0].meta["filename_or_obj"])
            plt.figure("check", (18, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image[:, :, 20], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask[0].detach().cpu()[1, :, :, 20])
            plt.savefig("TESTTTTT.png")

            # Compute metrics on preprocessed data
            dice_score = dice_metric(y_pred=pred_mask, y=gt_mask).item()
            hausdorff_distance = hausdorff_metric(y_pred=pred_mask, y=gt_mask).item()

            # # Resample to original spacing
            # pred_resampled = spacing_transform(pred_mask)
            # label_resampled = spacing_transform(gt_mask)

            # # Compute metrics on original spacing
            # iou, precision, recall, specificity = compute_additional_metrics(pred_resampled, label_resampled)

            # Log metrics
            metrics = {
                "Image Index": idx,
                "Dice Score": dice_score,
                "Hausdorff Distance": hausdorff_distance,
                # "IoU": iou,
                # "Precision": precision,
                # "Recall": recall,
                # "Specificity": specificity
            }
            log_metrics(metrics, log_file)

            #visualize_segmentation(original_image, pred_tensor, save_path)


if __name__ == "__main__":
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model_from_config('config/config.yaml').to(device)
    model = torch.nn.DataParallel(model)
    load_checkpoint(config['paths']['checkpoint'], model)
    model.eval()
    dataset_manager = DatasetManager(config)

    # Load test data
    test_loader = dataset_manager.get_dataloader("test")

    # Run inference
    run_inference(config, model, test_loader, dataset_manager.transforms, device)
