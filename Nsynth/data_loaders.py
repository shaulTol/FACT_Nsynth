import os
import subprocess
import torch
from torch.utils.data import DataLoader
from nsynth_dataset import NSynthDataset
from audio_augmentation import AudioFourierAugmentation

# Dataset Downloader
def download_nsynth_dataset():
    """Download and extract the NSynth dataset."""

    # Create directories
    os.makedirs('/content/nsynth', exist_ok=True)

    # Download datasets if they don't exist
    if not os.path.exists('/content/nsynth/nsynth-train'):
        print("Downloading training dataset...")
        subprocess.run(['wget', 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz', '-P', '/content'])
        print("Extracting training dataset...")
        subprocess.run(['tar', '-xf', '/content/nsynth-train.jsonwav.tar.gz', '-C', '/content/nsynth'])
    else:
        print("Training dataset already exists.")

    if not os.path.exists('/content/nsynth/nsynth-valid'):
        print("Downloading validation dataset...")
        subprocess.run(['wget', 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz', '-P', '/content'])
        print("Extracting validation dataset...")
        subprocess.run(['tar', '-xf', '/content/nsynth-valid.jsonwav.tar.gz', '-C', '/content/nsynth'])
    else:
        print("Validation dataset already exists.")

    if not os.path.exists('/content/nsynth/nsynth-test'):
        print("Downloading test dataset...")
        subprocess.run(['wget', 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz', '-P', '/content'])
        print("Extracting test dataset...")
        subprocess.run(['tar', '-xf', '/content/nsynth-test.jsonwav.tar.gz', '-C', '/content/nsynth'])
    else:
        print("Test dataset already exists.")

    print("Dataset download and extraction complete.")

# Domain Generalization Dataloaders
def get_fourier_train_dataloader(source_domains, batch_size=64, augmentation=True):
    """Create dataloaders for training with Fourier augmentation."""
    transform = AudioFourierAugmentation() if augmentation else None

    train_dataset = NSynthDataset(
        split='train',
        source_domains=source_domains,
        target_domain=None,
        transform=transform,
        max_samples_per_class=2000  # Limit samples for each class
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return train_loader, train_dataset.num_classes

def get_eval_dataloader(target_domain, split='valid', batch_size=64):
    """Create dataloaders for evaluation."""
    eval_dataset = NSynthDataset(
        split=split,
        source_domains=None,
        target_domain=target_domain,
        transform=None
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return eval_loader

######ADDED FUNCTION FIXING ISSUE##############
def get_split_dataloaders(source_domains, target_domain, config):
    """Create train, validation and test dataloaders with proper source/target split."""
    # Create a combined source dataset
    source_dataset = NSynthDataset(
        split='train',
        source_domains=source_domains,
        target_domain=None,
        max_samples_per_class=config.get('max_samples_per_class', 1000)
    )
    
    # Split source data into train and validation (80/20)
    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        source_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a wrapper for train dataset to apply augmentation
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
            
        def __getitem__(self, idx):
            data, target, domain = self.dataset[idx]
            if self.transform:
                data = self.transform(data)
            return data, target, domain
            
        def __len__(self):
            return len(self.dataset)
    
    # Apply Fourier augmentation to training data
    from audio_augmentation import AudioFourierAugmentation
    train_dataset = TransformDataset(
        train_dataset, 
        AudioFourierAugmentation() if config.get('use_augmentation', True) else None
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=4
    )
    
    # Test loader uses target domain
    test_loader = torch.utils.data.DataLoader(
        NSynthDataset(
            split='test',
            source_domains=None,
            target_domain=target_domain
        ),
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=4
    )
    
    # Get number of classes from the source dataset
    num_classes = source_dataset.num_classes
    
    return train_loader, val_loader, test_loader, num_classes

