import os
import tarfile
import gdown
import yaml
import nibabel as nib  # For 3D medical images

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

def download_dataset(gdrive_link, output_path):
    """Download dataset from Google Drive."""
    gdown.download(gdrive_link, output_path, quiet=False)

def extract_tar_file(tar_path, extract_to):
    """Extract tar file to the specified directory."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted tar file to {extract_to}")

def process_3d_data(data_dir):
    """Process extracted 3D data files."""
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                volume = nib.load(file_path)  # Load 3D volume
                print(f"Loaded volume: {file_path}, Shape: {volume.shape}")
                # Add further processing if required

def ingest_data():
    config = load_config()
    
    # Paths from config
    raw_data_dir = config["paths"]["raw_data"]
    extracted_data_dir = config["paths"]["extracted_data"]
    tar_file_path = os.path.join(raw_data_dir, config["parameters"]["tar_file_name"])
    
    # Step 1: Download the tar file
    if not os.path.exists(tar_file_path):
        print("Downloading dataset...")
        download_dataset(config["parameters"]["google_drive_link"], tar_file_path)
    
    # Step 2: Extract the tar file
    print("Extracting dataset...")
    extract_tar_file(tar_file_path, extracted_data_dir)
    
    # Step 3: Process 3D volumetric data
    print("Processing 3D volumetric data...")
    process_3d_data(extracted_data_dir)

if __name__ == "__main__":
    ingest_data()
