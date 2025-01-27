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
from utils import load_checkpoint, Config
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score
from monai.handlers.utils import from_engine

def enable_mc_dropout(model):
    """Enable MC Dropout by setting all dropout layers to train mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout3d):
            module.train()

def run_inference_with_mc_dropout(config, model, test_loader, transforms, device, mc_samples=10):
    """Runs inference with MC Dropout, computes metrics, and logs results."""
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

    roi_size = config['validation'].get('roi_size', (96, 96, 96))
    sw_batch_size = config['validation'].get('sw_batch_size', 4)

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            original_image = images[0, 0].cpu().numpy() 

            enable_mc_dropout(model)  
            mc_preds = torch.stack(
                [sliding_window_inference(images, roi_size, sw_batch_size, model) for _ in range(mc_samples)]
            )
            mc_mean = mc_preds.mean(dim=0)  
            mc_variance = mc_preds.var(dim=0) 

            batch_data["pred"] = mc_mean
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

            pred_mask, gt_mask = from_engine(["pred", "label"])(batch_data)

            dice_score = dice_metric(y_pred=pred_mask, y=gt_mask).item()
            hausdorff_distance = hausdorff_metric(y_pred=pred_mask, y=gt_mask).item()

            metrics = {
                "Image Index": idx,
                "Dice Score": dice_score,
                "Hausdorff Distance": hausdorff_distance,
            }
            with open(log_file, "a") as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

            pred_loader = LoadImage()
            original_image = pred_loader(pred_mask[0].meta["filename_or_obj"])
            plt.figure("check", (18, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image[:, :, 20], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(mc_mean[0, 1, :, :, 20].cpu().numpy(), cmap="jet", alpha=0.5)  # Overlay uncertainty
            plt.savefig(f"./out/mc_visual_{idx}.png")

            print(f"Image {idx}: Dice Score = {dice_score}, Hausdorff Distance = {hausdorff_distance}")

if __name__ == "__main__":

    config = Config("config/config.yaml")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_config('config/config.yaml').to(device)
    model = torch.nn.DataParallel(model)
    load_checkpoint(config['paths']['checkpoint'], model)

    model.eval()

    dataset_manager = DatasetManager(config)
    test_loader = dataset_manager.get_dataloader("test")

    run_inference_with_mc_dropout(config, model, test_loader, dataset_manager.transforms, device)
