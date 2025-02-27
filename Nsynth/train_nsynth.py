# train_nsynth.py
import argparse
import torch
import os
import sys

# Add the current directory to path for imports
sys.path.append('/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth')

from nsynth_trainer import NSynthDomainTrainer
from data_loaders import download_nsynth_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs='+', required=True, help="Source domains")
    parser.add_argument("--target", required=True, help="Target domain")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output_dir", default="/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/models", help="Output directory")
    return parser.parse_args()

def main():
    args = get_args()

    # First download and extract the dataset if needed
    download_nsynth_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'temperature': 2.0,
        'consistency_weight': 0.1,
        'epochs': args.epochs
    }

    print(f"Training on domains: {args.source}")
    print(f"Target domain: {args.target}")

    trainer = NSynthDomainTrainer(
        source_domains=args.source,
        target_domain=args.target,
        device=device,
        config=config
    )

    test_acc = trainer.train(
        epochs=args.epochs,
        save_dir=args.output_dir
    )

    print(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()