#config_utils.py
import yaml
import argparse
import os

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded from {config_path}")
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DistilBERT Fine-tuning/Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base_config.yaml",
        help="Path to the YAML configuration file"
    )
    # Allow overriding specific params via CLI (basic implementation)
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    
    return parser.parse_args()

def get_config():
    """Main entry point to get configuration."""
    args = parse_args()
    config = load_config(args.config)
    
    # Override if specified
    if args.output_dir:
        config["output_dir"] = args.output_dir
        
    # Ensure output dir exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    return config
