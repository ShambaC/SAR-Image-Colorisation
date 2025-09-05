import argparse
import yaml
import train_vae
import train_diffusion

def main():
    parser = argparse.ArgumentParser(description="Train a Latent Diffusion Model.")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['vae', 'diffusion'],
        help="Which model component to train: 'vae' or 'diffusion'."
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.model == 'vae':
        train_vae.train(config)
    elif args.model == 'diffusion':
        train_diffusion.train(config)
    else:
        print("Invalid model choice. Please choose 'vae' or 'diffusion'.")

if __name__ == '__main__':
    main()