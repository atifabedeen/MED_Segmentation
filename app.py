import streamlit as st
import torch
import tempfile
from monai.transforms import (
    Compose, Invertd, AsDiscreted, LoadImaged, Spacingd,
    EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd, LoadImage
)
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.handlers.utils import from_engine
from scripts.data_preprocessing import Config
from scripts.model_loader import load_model_from_config
from scripts.utils import load_checkpoint
import matplotlib.pyplot as plt
import os


def get_infer_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
    ])


def create_dataloader(file_path, transforms):
    data = [{"image": file_path}]
    dataset = Dataset(data=data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader


def run_inference(model, dataloader, post_transforms, roi_size, device):
    model.eval()
    pred_loader = LoadImage()

    with torch.no_grad():
        for batch_data in dataloader:
            images = batch_data["image"].to(device)

            predictions = sliding_window_inference(images, roi_size, 4, model)
            batch_data["pred"] = predictions
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

            pred_mask = from_engine(["pred"])(batch_data)
            pred_mask_tensor = torch.cat(pred_mask, dim=0)

            original_images = pred_loader(pred_mask[0].meta["filename_or_obj"])
    return pred_mask_tensor, original_images


def load_and_cache_model(config_path, checkpoint_path, device):
    config = Config(config_path)
    model = load_model_from_config(config_path).to(device)
    model = torch.nn.DataParallel(model)
    load_checkpoint(checkpoint_path, model)
    return model

def visualize_slices_streamlit(image, predictions):
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0

    depth = image.shape[2]
    slice_idx = st.slider(
        "Select Slice",
        min_value=0,
        max_value=depth - 1,
        value=st.session_state.slice_idx,
        key="slice_slider",
    )

    st.session_state.slice_idx = slice_idx

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image[:, :, slice_idx], cmap="gray")
    axes[0].set_title(f"Input Image (Slice {slice_idx})")

    axes[1].imshow(image[:, :, slice_idx], cmap="gray")
    axes[1].imshow(
        torch.argmax(predictions, dim=0).detach().cpu()[:, :, slice_idx],
        cmap="jet",
        alpha=0.5,
    )
    axes[1].set_title(f"Prediction Overlay (Slice {slice_idx})")

    st.pyplot(fig)
    plt.close(fig)

def main():
    st.title("3D MRI Segmentation Inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = {
        "VNET": ("config/config_VNET.yaml", "checkpoints/best_model_vnet-4.pth"),
        "UNET": ("config/config.yaml", "checkpoints/best_model_unet3d-2.pth"),
        "UNETR": ("config/config_UNETR.yaml", "checkpoints/best_model_unetr-4.pth"),
    }

    model_selected = None
    for model_name in model_paths:
        if st.button(f"Select {model_name}"):
            model_selected = model_name

    if model_selected:
        config_path, checkpoint_path = model_paths[model_selected]
        if "model" not in st.session_state or st.session_state.model_selected != model_selected:
            st.text(f"Loading {model_selected} model...")
            st.session_state.model = load_and_cache_model(config_path, checkpoint_path, device)
            st.session_state.transforms = get_infer_transforms()
            st.session_state.model_selected = model_selected
            st.session_state.inference_done = False

    uploaded_file = st.file_uploader("Upload a 3D MRI file (NIfTI format)", type=["nii", "nii.gz"])
    if uploaded_file and "model" in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name

        if (
            not st.session_state.get("inference_done", False)
            or st.session_state.file_path != uploaded_file.name
            or st.session_state.model_selected != model_selected
        ):
            st.text("Running inference...")
            post_transforms = Compose([
                Invertd(
                    keys="pred",
                    transform=st.session_state.transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                    device="cpu",
                ),
                AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            ])
            roi_size = [96, 96, 96]

            dataloader = create_dataloader(file_path, st.session_state.transforms)
            predictions, og_images = run_inference(
                st.session_state.model, dataloader, post_transforms, roi_size, device
            )

            st.session_state.predictions = predictions
            st.session_state.og_images = og_images
            st.session_state.file_path = uploaded_file.name
            st.session_state.inference_done = True

        st.success("Inference completed!")

        visualize_slices_streamlit(st.session_state.og_images, st.session_state.predictions)



if __name__ == "__main__":
    main()
