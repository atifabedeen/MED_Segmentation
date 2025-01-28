from monai.networks.nets import UNet, UNETR, VNet
from scripts.utils import Config

def load_model_from_config(config_path):
    """Load and initialize a model based on the configuration file, with added dropout for MC Dropout.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        nn.Module: Initialized model.
    """
    config = Config(config_path)
    model_name = config['model']['name']
    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    features = config['model'].get('features', None)
    strides = config['model'].get('strides', None)
    dropout_rate = config['model'].get('dropout_rate', 0.1) 

    if model_name == 'UNet3D':
        return UNet(
            spatial_dims=3, 
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides or [2] * (len(features) - 1),
            num_res_units=2,
            norm='batch',
            dropout=dropout_rate 
        )
    elif model_name == 'UNETR':
        return UNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=config['model'].get('img_size', [96, 96, 96]),
            feature_size=config['model'].get('feature_size', 16),
            hidden_size=config['model'].get('hidden_size', 768),
            mlp_dim=config['model'].get('mlp_dim', 3072),
            num_heads=config['model'].get('num_heads', 12),
            norm_name=config['model'].get('norm_name', 'instance'),
            dropout_rate=dropout_rate  
        )
    elif model_name == 'VNET':
        return VNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob_up=config['model'].get('dropout_prob_up', [dropout_rate, dropout_rate]),
            dropout_prob_down=config['model'].get('dropout_prob_down', dropout_rate)
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported by this loader.")
