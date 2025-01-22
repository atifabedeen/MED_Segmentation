from monai.networks.nets import UNet, UNETR, DynUNet
import yaml

def load_model_from_config(config_path):
    """Load and initialize a model based on the configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        nn.Module: Initialized model.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config['model']['name']
    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    features = config['model'].get('features', None)
    strides = config['model'].get('strides', None)

    if model_name == 'UNet3D':
        return UNet(
            spatial_dims=3, 
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides or [2] * (len(features) - 1),
            num_res_units=2,
            norm='batch'
        )
    elif model_name == 'UNETR':
        return UNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=config['model'].get('img_size', [128, 128, 64]),
            feature_size=config['model'].get('feature_size', 16),
            hidden_size=config['model'].get('hidden_size', 768),
            mlp_dim=config['model'].get('mlp_dim', 3072),
            num_heads=config['model'].get('num_heads', 12),
            pos_embed=config['model'].get('pos_embed', 'perceptron'),
            norm_name=config['model'].get('norm_name', 'instance'),
            dropout_rate=config['model'].get('dropout_rate', 0.0),
        )
    elif model_name == 'DynUNet':
        return DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=config['model'].get('kernel_size', [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]),
            strides=strides or [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=config['model'].get('upsample_kernel_size', [[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
            filters=features,
            norm_name=config['model'].get('norm_name', 'instance'),
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported by this loader.")
