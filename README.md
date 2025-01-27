# MED_Segmentation

## Overview
**MED_Segmentation** is a medical image segmentation pipeline designed for segmenting 3D volumetric data, particularly for Left Atrium segmentation from MRI images. The pipeline implements state-of-the-art deep learning models such as **UNETR** and **VNET**, and includes steps for data preprocessing, training, evaluation, and uncertainty quantification. 

Key features:
- Data preprocessing with augmentations
- Support for multiple model architectures (UNETR, VNET)
- Experiment tracking using MLFlow
- Version control with DVC
- Uncertainty quantification using methods like Monte Carlo Dropout

---

## File Structure

### Main Scripts
| File | Description |
|------|-------------|
| `main.py` | Entry point for running the full pipeline. Sequentially executes all steps including data ingestion, preprocessing, training, and evaluation. |
| `data_ingestion.py` | Handles data downloading, extraction, and organization. Reads from `config.yaml` for source and destination paths. |
| `data_preprocessing.py` | Prepares data for training, including resizing, normalization, and augmentation based on parameters in `config.yaml`. |
| `model_loader.py` | Defines and loads model architectures (e.g., UNETR, VNET). Provides a modular approach for model configuration. |
| `model_training.py` | General script for training models. Accepts model, dataset, and architecture parameters to initiate training. |
| `training_UNETR.py` | Specialized script for training the UNETR architecture. |
| `training_VNET.py` | Specialized script for training the VNET architecture. |
| `model_evaluation_mc.py` | Evaluates trained models using metrics and implements Monte Carlo Dropout for uncertainty quantification. |
| `utils.py` | Utility functions used across the pipeline for common tasks like file handling, logging, and metric calculations. |

### Configuration Files
| File | Description |
|------|-------------|
| `config.yaml` | Contains configuration parameters such as paths, preprocessing options, training hyperparameters, and logging settings. |
| `dataset.json` | Metadata describing the dataset structure, labels, and file paths for training and testing. |

### DVC Files
| File | Description |
|------|-------------|
| `dvc.yaml` | Defines DVC stages for data ingestion, preprocessing, training, and evaluation. Ensures reproducibility and version control for the pipeline. |

---

## Setup Instructions

### Prerequisites
Ensure the following dependencies are installed:
- Python 3.8+
- Pytorch
- SimpleITK
- MLFlow
- DVC
- Streamlit / Gradio (optional for UI)

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Dataset Preparation
1. Modify `config.yaml`:
   - Set the `raw_data` path for where the raw data will be downloaded.
   - Define the `extracted_data` path for storing extracted files.
2. Run the data ingestion script:
   ```bash
   python scripts/data_ingestion.py
   ```

---

## Steps to Run the Pipeline

### 1. Data Ingestion
Download and organize the dataset. Run:
```bash
python scripts/data_ingestion.py
```

### 2. Data Preprocessing
Preprocess the dataset by resizing, normalizing, and augmenting images:
```bash
python scripts/data_preprocessing.py
```

### 3. Train the Models
#### Train UNETR:
```bash
python scripts/training_UNETR.py
```

#### Train VNET:
```bash
python scripts/training_VNET.py
```

### 4. Evaluate Models
Evaluate the trained models and compute metrics such as Dice Score, Jaccard Index, etc.:
```bash
python scripts/model_evaluation_mc.py
```

### 5. Run the Full Pipeline
To run all the steps sequentially:
```bash
python main.py
```

---

## Logging and Experiment Tracking
- **MLFlow** is integrated for tracking experiments. Start the MLFlow UI by running:
  ```bash
  mlflow ui
  ```
  Navigate to `http://localhost:5000` to view logged metrics, parameters, and model checkpoints.

---

## Uncertainty Quantification
Monte Carlo Dropout is used to estimate uncertainty in predictions. The script `model_evaluation_mc.py` includes functionality for this analysis.

---

## Version Control with DVC
1. Initialize DVC in the repository:
   ```bash
   dvc init
   ```
2. Add files to DVC tracking (e.g., datasets):
   ```bash
   dvc add data/raw
   ```
3. Commit changes:
   ```bash
   git add .
   git commit -m "Track data with DVC"
   ```
4. Push changes to remote storage:
   ```bash
   dvc push
   ```

---

## UI for Visualization
(Optional) Build a user interface for model inference using Streamlit or Gradio. Customize the UI to load an MRI image and display segmentation results.

Run the UI:
```bash
streamlit run ui.py
```

---

## Acknowledgments
- Dataset: Left Atrium segmentation dataset from [Medical Segmentation Decathlon](http://medicaldecathlon.com).
- Reference frameworks: PyTorch, MLFlow, and DVC.
