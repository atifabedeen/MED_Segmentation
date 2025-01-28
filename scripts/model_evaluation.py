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
from scripts.data_preprocessing import DatasetManager, Config, get_transforms
from scripts.model_loader import load_model_from_config
from scripts.utils import load_checkpoint, Config
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score
from monai.handlers.utils import from_engine

CONFIG_FILE_PATH = "config/config.yaml"

def compute_metrics(preds, labels):
    """Computes IoU, Precision, Recall, and Specificity."""
    preds_flat = torch.cat(preds).flatten().cpu().numpy()  # Convert list to tensor, flatten, and convert to numpy
    labels_flat = torch.cat(labels).flatten().cpu().numpy()

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

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(percentile=95, include_background=False)
    #spacing_transform = Spacing(pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"))

    roi_size = config['validation'].get('roi_size', (96, 96, 96))
    sw_batch_size = config['validation'].get('sw_batch_size', 4)
    pred_loader = LoadImage()

    dice_scores = []
    hausdorff_distances = []
    ious = []
    precisions = []
    recalls = []
    specificities = []

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            original_image = images[0, 0].cpu().numpy()  

            batch_data["pred"] = sliding_window_inference(images, roi_size, sw_batch_size, model)
            #val_outputs = sliding_window_inference(images, roi_size, sw_batch_size, model)
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

            pred_mask, gt_mask = from_engine(["pred", "label"])(batch_data)
            gt_mask_tensor = torch.cat(gt_mask, dim=0)
            pred_mask_tensor = torch.cat(pred_mask, dim=0)
            original_image = pred_loader(pred_mask[0].meta["filename_or_obj"])
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(images[0, 0, :, :, 20].cpu().numpy(), cmap="gray")
            plt.subplot(1, 3, 2)
            plt.imshow(labels[0, 0, :, :, 20].cpu().numpy())
            plt.subplot(1, 3, 3)
            plt.imshow(torch.argmax(pred_mask_tensor, dim=0).detach().cpu()[:, :, 20])
            plt.savefig("Test.png")

            dice_score = dice_metric(y_pred=pred_mask, y=gt_mask).item()
            hausdorff_distance = hausdorff_metric(y_pred=pred_mask, y=gt_mask).item()

            iou, precision, recall, specificity = compute_metrics(pred_mask, gt_mask)

            dice_scores.append(dice_score)
            hausdorff_distances.append(hausdorff_distance)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)

            metrics = {
                "Image Index": idx,
                "Dice Score": dice_score,
                "Hausdorff Distance": hausdorff_distance,
                "IoU": iou,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity
            }
            log_metrics(metrics, log_file)

            avg_metrics = {
                "Average Dice Score": sum(dice_scores) / len(dice_scores),
                "Average Hausdorff Distance": sum(hausdorff_distances) / len(hausdorff_distances),
                "Average IoU": sum(ious) / len(ious),
                "Average Precision": sum(precisions) / len(precisions),
                "Average Recall": sum(recalls) / len(recalls),
                "Average Specificity": sum(specificities) / len(specificities)
            }

            log_metrics(avg_metrics, log_file)

def main():
    config = Config(CONFIG_FILE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_config('config/config.yaml').to(device)
    model = torch.nn.DataParallel(model)
    load_checkpoint(os.path.join("checkpoints", f"best_model_{config['model']['name'].lower()}.pth"), model)
    model.eval()
    dataset_manager = DatasetManager(config)

    test_loader = dataset_manager.get_dataloader("test")

    run_inference(config, model, test_loader, dataset_manager.transforms, device)

if __name__ == "__main__":
    main()