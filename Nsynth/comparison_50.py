import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add necessary paths
sys.path.append('/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth')

# Import the trainers
from nsynth_trainer import NSynthDomainTrainer
from deep_all_trainer import DeepAllTrainer
from data_loaders import download_nsynth_dataset

def run_comparison(target_domain, epochs=50, max_samples=1000, batch_size=64):
    """
    Run FACT and DeepAll comparison for a specified target domain.
    
    Args:
        target_domain: Target domain to evaluate on (acoustic, electronic, or synthetic)
        epochs: Number of epochs to train for
        max_samples: Maximum samples per class per domain
        batch_size: Batch size for training
    """
    # First download and extract the dataset if needed
    download_nsynth_dataset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define domains
    all_domains = ['acoustic', 'electronic', 'synthetic']
    
    # Verify target domain is valid
    if target_domain not in all_domains:
        raise ValueError(f"Target domain must be one of {all_domains}")
        
    # Get source domains
    source_domains = [d for d in all_domains if d != target_domain]
    
    # Common configuration for both methods
    config = {
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'temperature': 2.0,
        'consistency_weight': 0.1,
        'max_samples_per_class': max_samples,
        'epochs': epochs,
        'use_augmentation': True
    }
    
    # Create directories for saving models
    fact_dir = f'/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/fact_models_{target_domain}'
    deepall_dir = f'/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/deepall_models_{target_domain}'
    os.makedirs(fact_dir, exist_ok=True)
    os.makedirs(deepall_dir, exist_ok=True)
    
    # Results dictionary
    results = {
        'fact': {},
        'deepall': {},
        'target_domain': target_domain,
        'source_domains': source_domains,
        'config': config
    }
    
    print(f"\n{'='*70}")
    print(f"Target domain: {target_domain}")
    print(f"Source domains: {source_domains}")
    print(f"{'='*70}\n")
    
    # FACT training
    print("\nRunning FACT (Fourier-based Domain Generalization)...")
    fact_trainer = NSynthDomainTrainer(
        source_domains=source_domains,
        target_domain=target_domain,
        device=device,
        config=config
    )
    
    fact_test_acc = fact_trainer.train(
        epochs=config['epochs'],
        save_dir=fact_dir
    )
    
    # DeepAll training
    print("\nRunning DeepAll baseline...")
    deepall_trainer = DeepAllTrainer(
        source_domains=source_domains,
        target_domain=target_domain,
        device=device,
        config=config
    )
    
    deepall_test_acc = deepall_trainer.train(
        epochs=config['epochs'],
        save_dir=deepall_dir
    )
    
    # Save results for this domain
    results['fact']['test_accuracy'] = fact_test_acc
    results['deepall']['test_accuracy'] = deepall_test_acc
    
    # Print comparison for this domain
    print(f"\n{'-'*70}")
    print(f"RESULTS FOR TARGET DOMAIN: {target_domain}")
    print(f"FACT Test Accuracy: {fact_test_acc:.2f}%")
    print(f"DeepAll Test Accuracy: {deepall_test_acc:.2f}%")
    print(f"Improvement: {fact_test_acc - deepall_test_acc:.2f}%")
    print(f"{'-'*70}\n")
    
    # Save results to file
    results_file = f'/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/{target_domain}_comparison_50ep.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a bar chart comparing the results
    create_comparison_chart(fact_test_acc, deepall_test_acc, target_domain)
    
    return results

def create_comparison_chart(fact_acc, deepall_acc, target_domain):
    """Create a bar chart comparing FACT and DeepAll results."""
    plt.figure(figsize=(10, 6))
    
    methods = ['DeepAll', 'FACT']
    accuracies = [deepall_acc, fact_acc]
    colors = ['skyblue', 'salmon']
    
    plt.bar(methods, accuracies, color=colors)
    plt.axhline(y=deepall_acc, color='gray', linestyle='--', alpha=0.7)
    
    # Add improvement annotation
    improvement = fact_acc - deepall_acc
    plt.annotate(f'+{improvement:.2f}%', 
                xy=(1, fact_acc),
                xytext=(1, fact_acc + 2),
                ha='center',
                va='bottom',
                fontsize=12,
                color='green' if improvement > 0 else 'red')
    
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'FACT vs DeepAll: Target Domain = {target_domain} (50 epochs)')
    
    # Add value labels on bars
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.2f}%', 
                    xy=(i, acc), 
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'/content/AFourier-based-Framework-for-Domain-Generalization/Nsynth/{target_domain}_comparison_50ep.png')
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run domain generalization comparison')
    parser.add_argument('--target', type=str, required=True, choices=['acoustic', 'electronic', 'synthetic'],
                        help='Target domain to evaluate on')
    
    args = parser.parse_args()
    
    # Run the comparison with the specified target domain
    run_comparison(
        target_domain=args.target,
        epochs=50,
        max_samples=1000,
        batch_size=64
    )