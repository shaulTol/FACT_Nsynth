import torch
import numpy as np
import json
import os
import sys

# Add necessary paths
sys.path.append('/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth')

from deep_all_trainer import DeepAllTrainer
from data_loaders import download_nsynth_dataset

def run_deep_all_leave_one_out():
    """Run leave-one-domain-out training for DeepAll approach."""

    # First download and extract the dataset if needed
    download_nsynth_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define domains
    all_domains = ['acoustic', 'electronic', 'synthetic']

    # Config
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'max_samples_per_class': 2000,
        'epochs': 50
    }

    # Results
    results = {}

    # For each target domain, train on remaining domains
    for target_domain in all_domains:
        source_domains = [d for d in all_domains if d != target_domain]

        print(f"\n{'='*50}")
        print(f"Target domain: {target_domain}")
        print(f"Source domains: {source_domains}")
        print(f"{'='*50}\n")

        trainer = DeepAllTrainer(
            source_domains=source_domains,
            target_domain=target_domain,
            device=device,
            config=config
        )

        test_acc = trainer.train(
            epochs=config['epochs'],
            save_dir='/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/deepall_models'
        )

        results[target_domain] = {
            'source_domains': source_domains,
            'test_accuracy': test_acc
        }

    # Print and save overall results
    print("\nOverall Results:")
    avg_acc = 0
    for target, result in results.items():
        acc = result['test_accuracy']
        avg_acc += acc
        print(f"Target: {target}, Accuracy: {acc:.2f}%")

    avg_acc /= len(all_domains)
    print(f"\nAverage Accuracy: {avg_acc:.2f}%")

    # Save results
    with open('/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/deepall_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_deep_all_leave_one_out()