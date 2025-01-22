from monai.networks.nets import UNet
import yaml

def load_model_from_config(config_path):
    """Load and initialize a model based on the configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        nn.Module: Initialized model.
    """
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['name']
    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    features = config['model']['features']

    if model_name == 'UNet3D':
        return UNet(
            spatial_dims=3,  # Specify spatial dimensions
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=[2] * (len(features) - 1),
            num_res_units=2,
            norm='batch'
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported by this loader.")
