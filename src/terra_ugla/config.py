"""
Configuration management for TERRA UGLA
"""

import json
import os
from sentinelhub.config import SHConfig


def load_sentinel_hub_config():
    """Load Sentinel Hub configuration from config.json file"""
    config_file = 'config.json'

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found. "
            f"Please copy 'config.example.json' to '{config_file}' and add your Sentinel Hub credentials."
        )

    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        sentinel_config = config_data.get('sentinel_hub', {})

        if not sentinel_config.get('client_id') or sentinel_config.get('client_id') == 'your_client_id_here':
            raise ValueError("Please set your Sentinel Hub client_id in config.json")

        if not sentinel_config.get('client_secret') or sentinel_config.get('client_secret') == 'your_client_secret_here':
            raise ValueError("Please set your Sentinel Hub client_secret in config.json")

        return sentinel_config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config.json: {e}")
    except Exception as e:
        raise Exception(f"Error loading configuration: {e}")


def initialize_sentinel_hub_config():
    """
    Initialize and return Sentinel Hub configuration
    Returns tuple: (config, is_available)
    """
    config = SHConfig()
    sentinel_hub_available = False

    try:
        sentinel_config = load_sentinel_hub_config()

        config = SHConfig()

        # CRITICAL: Set ALL the required URLs for CDSE
        config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
        config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        config.sh_auth_base_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect'

        # THEN set credentials
        config.sh_client_id = sentinel_config['client_id']
        config.sh_client_secret = sentinel_config['client_secret']

        # DO NOT set instance_id for CDSE

        sentinel_hub_available = True
        print("Sentinel Hub configuration loaded successfully")
    except Exception as e:
        print(f"Warning: Sentinel Hub configuration error: {e}")
        print("Satellite data functionality will use demo mode")
        sentinel_hub_available = False

    return config, sentinel_hub_available


def create_data_directories():
    """Create necessary data directories"""
    os.makedirs('data/shorelines', exist_ok=True)
    os.makedirs('data/aoi', exist_ok=True)
    os.makedirs('data/baselines', exist_ok=True)
    os.makedirs('data/transects', exist_ok=True)
    os.makedirs('data/satellite_images', exist_ok=True)
    os.makedirs('data/runs', exist_ok=True)
    os.makedirs('data/dt', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/exports', exist_ok=True)
