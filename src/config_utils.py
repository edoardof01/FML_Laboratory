# config_utils.py
"""Centralized configuration loading from YAML + CLI overrides."""
import yaml
import argparse
import os


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded from {config_path}")
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DistilBERT Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    return parser.parse_args()


def get_config() -> dict:
    """Main entry point: parse CLI args → load YAML → apply overrides."""
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    os.makedirs(config["output_dir"], exist_ok=True)
    return config
