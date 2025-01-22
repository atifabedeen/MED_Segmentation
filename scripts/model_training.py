import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.losses import DiceLoss
from model_loader import load_model_from_config
from data_preprocessing import MRIDataset
from utils import save_checkpoint
import mlflow
import yaml

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = load_model_from_config('config/config.yaml').to(device)

# Define loss and optimizer
criterion = DiceLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# Load datasets
crop_dim = (
    config['preprocessing']['tile_height'],
    config['preprocessing']['tile_width'],
    config['preprocessing']['tile_depth']
)
data_train = MRIDataset(crop_dim, config, mode='train')
data_val = MRIDataset(crop_dim, config, mode='val')
train_loader = DataLoader(data_train, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = DataLoader(data_val, batch_size=config['training']['batch_size'], shuffle=False)

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    best_val_loss = float('inf')
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        mlflow.log_params({
            'model_name': config['model']['name'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': num_epochs
        })

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
                data, target = data.to(device), target.to(device)
                target = target.permute(0, 4, 1, 2, 3)
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
            mlflow.log_metric('train_loss', avg_train_loss, step=epoch)

            # Validation
            val_loss = validate(model, val_loader, criterion, epoch)
            print(f"Validation Loss: {val_loss:.4f}")
            mlflow.log_metric('val_loss', val_loss, step=epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, config['paths']['checkpoint'], epoch=epoch, val_loss=val_loss)
                mlflow.log_artifact(config['paths']['checkpoint'], artifact_path="model")



def validate(model, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation", leave=False):
            data, target = data.to(device), target.to(device)
            target = target.permute(0, 4, 1, 2, 3)

            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()

    return val_loss / len(val_loader)

if __name__ == "__main__":
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=config['training']['num_epochs']
    )
