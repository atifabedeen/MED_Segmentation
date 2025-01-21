# Directory and file setup
import os

# Define the directory structure
directory_structure = {
    "config": ["config.yaml"],
    "data": [],
    "scripts": [
        "data_ingestion.py",
        "data_preprocessing.py",
        "model_training.py",
        "model_evaluation.py",
        "utils.py",
    ],
    "notebooks": [],
    "tests": ["test_pipeline.py"],
    ".github/workflows": ["ci.yml"],
}

# Base files
base_files = ["requirements.txt", "README.md", ".gitignore", "main.py"]

# Recursive function to create directories and files
def create_directories_and_files(base_path, structure):
    for folder, content in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        if isinstance(content, dict):  # If the value is a dictionary, recurse
            create_directories_and_files(folder_path, content)
        else:
            for file_name in content:  # Otherwise, create files
                file_path = os.path.join(folder_path, file_name)
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write("# Placeholder\n")

# Create directories and files
def setup_project_structure():
    create_directories_and_files(".", directory_structure)

    # Create base files
    for file_path in base_files:
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("# Placeholder\n")

setup_project_structure()
