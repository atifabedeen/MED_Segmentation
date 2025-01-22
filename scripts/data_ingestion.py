import os
import tarfile
import gdown
import yaml
import nibabel as nib


def load_config():
    """Load the configuration from config.yaml."""
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)


def download_dataset(gdrive_link, output_path):
    """Download dataset from Google Drive if it doesn't already exist."""
    if not os.path.exists(output_path):
        print(f"Downloading dataset to {output_path}...")
        gdown.download(gdrive_link, output_path, quiet=False)
        print(f"Dataset downloaded to {output_path}")
    else:
        print(f"Dataset already exists at {output_path}. Skipping download.")


def extract_tar_file(tar_path, extract_to):
    """Extract tar file to the specified directory."""
    if not os.path.exists(extract_to) or not os.listdir(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        print(f"Extracting tar file {tar_path} to {extract_to}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_to)
        print(f"Extraction completed. Files are now in {extract_to}")
    else:
        print(f"Extracted data already exists at {extract_to}. Skipping extraction.")


def process_3d_data(data_dir):
    """Process extracted 3D data files."""
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                volume = nib.load(file_path)
                print(f"Loaded volume: {file_path}, Shape: {volume.shape}")


def ingest_data():
    """Main function to ingest data."""
    config = load_config()

    raw_data_dir = config["paths"]["raw_data"]
    extracted_data_dir = config["paths"]["extracted_data"]
    tar_file_path = os.path.join(raw_data_dir, config["parameters"]["tar_file_name"])

    download_dataset(config["parameters"]["google_drive_link"], tar_file_path)

    extract_tar_file(tar_file_path, extracted_data_dir)

    process_3d_data(extracted_data_dir)


if __name__ == "__main__":
    ingest_data()
