import streamlit as st
import torch
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, Invertd, AsDiscreted, SaveImaged
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine
from scripts.data_preprocessing import get_transforms, Config
from scripts.model_loader import load_model_from_config
from scripts.utils import load_checkpoint

@st.cache_resource
def load_model_from_checkpoint(config_path):
    config = Config(config_path)
    model = load_model_from_config(config_path).to("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model)
    load_checkpoint(config['paths']['checkpoint'], model)
    model.eval()
    return model, config

def preprocess_image(file_path, config):
    transforms = get_transforms(config, mode='infer')  
    data = {"image": file_path}
    transformed = transforms(data)
    return transformed["image"]

def get_post_transforms(transforms):
    return Compose([
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
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
    ])

def run_inference(model, image, roi_size, transforms, post_transforms):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0).to(device).clone().detach()
    print(f"Image tensor shape (before inference): {image_tensor.shape}")
    print(f"ROI size: {roi_size}")

    with torch.no_grad():
        predictions = sliding_window_inference(image_tensor, roi_size, 4, model)
    
    batch_data = {"pred": predictions, "image": image_tensor}
    batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]
    
    final_prediction = batch_data[0]["pred"].argmax(dim=0).cpu().numpy()
    return final_prediction


def visualize_slices(image, prediction, slice_idx):
    image = image.squeeze()  
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image[:, :, slice_idx], cmap="gray")
    axes[0].set_title("Input Image")

    axes[1].imshow(image[:, :, slice_idx], cmap="gray")
    axes[1].imshow(prediction[:, :, slice_idx], cmap="jet", alpha=0.5)
    axes[1].set_title("Segmentation Overlay")

    st.pyplot(fig)


def main():
    st.title("3D MRI Segmentation Inference")
    
    config_path = "config/config.yaml"  
    st.sidebar.info(f"Using config file: {config_path}")
    model, config = load_model_from_checkpoint(config_path)
    uploaded_file = st.file_uploader("Upload a 3D MRI file (NIfTI format)", type=["nii", "nii.gz"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.text("Applying preprocessing transforms...")
        transforms = get_transforms(config, mode='infer')
        processed_image = preprocess_image(temp_file_path, config)
        processed_image = np.array(processed_image)  
        st.text(f"Processed image shape: {processed_image.shape}")

        post_transforms = get_post_transforms(transforms)

        st.text("Running inference...")
        roi_size = [96, 96, 96]
        prediction = run_inference(model, processed_image, roi_size, transforms, post_transforms)
        st.success("Inference completed!")

        st.subheader("Slice Visualization")

        num_slices = processed_image.shape[-1]  
        print(f"Number of slices: {num_slices}")  
        slice_idx = st.slider("Select Slice", 0, num_slices - 1, num_slices // 2)

        visualize_slices(processed_image, prediction, slice_idx)



if __name__ == "__main__":
    main()
