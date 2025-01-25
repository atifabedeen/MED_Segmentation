import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from monai.data import decollate_batch
from utils import save_checkpoint, log_to_mlflow, Config
from model_loader import load_model_from_config
from data_preprocessing import DatasetManager, Config
import mlflow
import os

# Reproducibility
set_determinism(seed=42)

config = Config("config/config.yaml")
dataset_manager = DatasetManager(config)

train_loader = dataset_manager.get_dataloader('train')
val_loader = dataset_manager.get_dataloader('val')

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config['model']['out_channels'])])
post_label = Compose([AsDiscrete(to_onehot=config['model']['out_channels'])])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.nn.DataParallel(load_model_from_config('config/config.yaml')).to(device)

criterion = DiceLoss(to_onehot_y=True, sigmoid=True)

dice_metric = DiceMetric(include_background=False, reduction="mean")

optimizer = optim.Adam(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training'].get('weight_decay', 1e-5)
)
scheduler = StepLR(optimizer, step_size=config['training']['scheduler_step'], gamma=config['training']['scheduler_gamma'])


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs):
    val_interval = config['training'].get('val_interval', 1)
    best_metric = -1
    best_metric_epoch = -1
    patience = config['training'].get('patience', 5)
    patience_counter = 0

    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        log_to_mlflow(config)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            model.train()
            epoch_loss = 0

            for batch_data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
                images, labels = batch_data["image"].to(device), batch_data["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            avg_train_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
            mlflow.log_metric('train_loss', avg_train_loss, step=epoch)

            if (epoch + 1) % val_interval == 0:
                avg_dice = validate(model, val_loader, dice_metric, device, config, epoch)
                print(f"Validation Dice: {avg_dice:.4f}")
                mlflow.log_metric('val_dice', avg_dice, step=epoch)

                if avg_dice > best_metric:
                    best_metric = avg_dice
                    best_metric_epoch = epoch + 1
                    config['paths']['checkpoint'] = os.path.join("checkpoints", f"best_model_{config['model']['name'].lower()}.pth")

                    save_checkpoint(model, optimizer, config['paths']['checkpoint'], epoch=epoch, val_loss=avg_dice)
                    print("Saved new best model checkpoint.")

                print(
                    f"Current Epoch: {epoch+1}, Best Metric: {best_metric:.4f} "
                    f"at Epoch: {best_metric_epoch}"
                )

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break


def validate(model, val_loader, dice_metric, device, config, epoch):
    """
    Validation function with sliding window inference, dice metric computation, and post-processing.
    """
    model.eval()
    dice_metric.reset()
    roi_size = config['validation'].get('roi_size', (96, 96, 96))
    sw_batch_size = config['validation'].get('sw_batch_size', 4)

    with torch.no_grad():
        for val_data in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            dice_metric(y_pred=val_outputs, y=val_labels)

        avg_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        print(f"Validation Dice for Epoch {epoch+1}: {avg_dice:.4f}")
        return avg_dice


if __name__ == "__main__":
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        num_epochs=config['training']['num_epochs']
    )
