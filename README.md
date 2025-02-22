# MED_Segmentation

## Overview
**MED_Segmentation** is a medical image segmentation pipeline designed for segmenting 3D volumetric data, particularly for Spleen segmentation from MRI images. The pipeline implements deep learning models such as **UNET**, **VNET**, and **UNETR**, and includes steps for data preprocessing, training, evaluation, and uncertainty quantification. 

Key features:
- Data preprocessing with augmentations
- Support for multiple model architectures (UNET, VNET, UNETR)
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
| `model_loader.py` | Defines and loads model architectures (e.g., UNET, VNET, UNETR). Provides a modular approach for model configuration. |
| `model_training.py` | General script for training models. Accepts model, dataset, and architecture parameters to initiate training. |
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

### 1. Setup
Helps setup the repository with any missing directories:
```
python setup.py
```

### 2. Data Ingestion
Download and organize the dataset. Run:
```bash
python -m scripts.data_ingestion
```

### 3. Data Preprocessing
Preprocess the dataset by resizing, normalizing, and augmenting images:
```bash
python -m scripts.data_preprocessing
```

### 4. Train the Models
```bash
python -m scripts.model_training
```
### 5. Evaluate Models
Evaluate the trained models and compute metrics such as Dice Score, Hausdorff Distance, Precision, and Recall:
```bash
python -m scripts.model_evaluation
```
### 6. Uncertainty Quantification
Evaluate the model using Mote Carlo Dropout and visualize the uncertainty maps:
```bash
python -m scripts.model_evaluation_mc
```

Metrics computed during evaluation:
- **Dice Score**: Measures the overlap between predicted and ground truth masks.
- **Hausdorff Distance (95th Percentile)**: Quantifies the boundary agreement between predictions and ground truth.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.

### 7. Run the Full Pipeline
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
Monte Carlo Dropout is used to estimate uncertainty in predictions. The script `model_evaluation_mc.py` includes functionality for this analysis. Uncertainty maps are saved as visualizations for further interpretation.

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
Built a user interface for model inference using Streamlit. Users can select between the three models, upload volumetric data (Nifti format) and visualize each slice of the volume with the segmentation overlayed on top of the input image. A slight warning: inferencing is a bit slow because I was having an issue with caching the inferneced data (it would get deleted when the page re-renders so it will take some time if you select a slice)

Please download the model weights from [this link](https://drive.google.com/drive/folders/155cLJLto8EIZunVQBtZb3bm0wxd30yF0?usp=drive_link) for the three models and store it in an appropriate directory. You will need to give the path to these three models in app.py

![UI](images/UI_new.png)


Run the UI:
```bash
streamlit run app.py
```

---

## Acknowledgments
- AI Usage: AI was used to generate documentation, debugging, and optimizing workflows.
- MONAI [ML Workflow example](https://github.com/Project-MONAI/tutorials/tree/main) for Medical Segmentation Decathlon was used a reference. 
- Dataset: Spleen segmentation dataset from [Medical Segmentation Decathlon](http://medicaldecathlon.com).
