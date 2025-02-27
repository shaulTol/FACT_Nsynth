import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from audio_resnet import AudioResNet18
from data_loaders import get_split_dataloaders, get_eval_dataloader, download_nsynth_dataset


class NSynthDomainTrainer:
    def __init__(self, source_domains, target_domain, device, config):
        """
        Args:
            source_domains: List of source domains for training
            target_domain: Target domain for testing
            device: Device to run on (cuda/cpu)
            config: Configuration dictionary
        """
        self.source_domains = source_domains
        self.target_domain = target_domain
        self.device = device
        self.config = config

        # Get dataloaders with proper split
        self.train_loader, self.val_loader, self.test_loader, num_classes = get_split_dataloaders(
            source_domains,
            target_domain,
            config
        )


        self.val_loader = get_eval_dataloader(
            target_domain,
            split='valid',
            batch_size=config.get('batch_size', 64)
        )

        self.test_loader = get_eval_dataloader(
            target_domain,
            split='test',
            batch_size=config.get('batch_size', 64)
        )

        # Initialize models
        self.encoder = AudioResNet18(num_classes=num_classes).to(device)

        # Initialize teacher models
        self.encoder_teacher = AudioResNet18(num_classes=num_classes).to(device)
        self.copy_model_params(self.encoder, self.encoder_teacher)

        # Initialize optimizers
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()),
            lr=config.get('learning_rate', 0.001)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.global_step = 0

    def copy_model_params(self, source, target):
        """Copy parameters from source model to target model."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
            target_param.requires_grad = False

    def update_teacher(self, alpha=0.999):
        """Update teacher model using EMA."""
        for teacher_param, student_param in zip(self.encoder_teacher.parameters(), self.encoder.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1-alpha)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.encoder.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets, domains) in enumerate(tqdm(self.train_loader)):
            # Move to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Split batch into original and augmented parts
            # In the original and augmented framework, we'd have augmented versions
            # Here we'll use our Fourier augmentation on half the batch
            batch_size = inputs.size(0)
            half_batch = batch_size // 2

            inputs_orig = inputs[:half_batch]
            inputs_aug = inputs[half_batch:]
            targets_orig = targets[:half_batch]
            targets_aug = targets[half_batch:]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass - original data
            outputs_orig, features_orig = self.encoder(inputs_orig)
            loss_orig = self.criterion(outputs_orig, targets_orig)

            # Forward pass - augmented data
            outputs_aug, features_aug = self.encoder(inputs_aug)
            loss_aug = self.criterion(outputs_aug, targets_aug)

            # Forward pass with teacher model (no gradients)
            with torch.no_grad():
                outputs_orig_teacher, features_orig_teacher = self.encoder_teacher(inputs_orig)
                outputs_aug_teacher, features_aug_teacher = self.encoder_teacher(inputs_aug)

            # Consistency loss (using KL divergence)
            temperature = self.config.get('temperature', 2.0)

            p_orig = F.softmax(outputs_orig / temperature, dim=1)
            p_aug = F.softmax(outputs_aug / temperature, dim=1)
            p_orig_teacher = F.softmax(outputs_orig_teacher / temperature, dim=1)
            p_aug_teacher = F.softmax(outputs_aug_teacher / temperature, dim=1)

            loss_consist_orig = F.kl_div(p_aug.log(), p_orig_teacher, reduction='batchmean')
            loss_consist_aug = F.kl_div(p_orig.log(), p_aug_teacher, reduction='batchmean')

            # Total loss
            consist_weight = min(1.0, epoch / 10.0) * self.config.get('consistency_weight', 0.1)
            total_loss = loss_orig + loss_aug + consist_weight * (loss_consist_orig + loss_consist_aug)

            # Backward and optimize
            total_loss.backward()
            self.optimizer.step()

            # Update teacher model
            self.update_teacher()

            # Statistics
            running_loss += total_loss.item()
            _, predicted = outputs_orig.max(1)
            total += targets_orig.size(0)
            correct += predicted.eq(targets_orig).sum().item()

            self.global_step += 1

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total

        return train_loss, train_acc

    def evaluate(self, dataloader):
        """Evaluate the model on the given dataloader."""
        self.encoder.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, _ in tqdm(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, _ = self.encoder(inputs)
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def train(self, epochs=50, save_dir='./checkpoints'):
        """Train for specified number of epochs."""
        os.makedirs(save_dir, exist_ok=True)

        print(f"Training on domains: {self.source_domains}")
        print(f"Target domain: {self.target_domain}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_acc = self.evaluate(self.val_loader)

            print(f"Epoch: {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = os.path.join(save_dir, f"best_model_{'-'.join(self.source_domains)}_to_{self.target_domain}.pth")
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                }, save_path)
                print(f"Model saved to {save_path}")

            print("-" * 50)

        # Test with the best model
        save_path = os.path.join(save_dir, f"best_model_{'-'.join(self.source_domains)}_to_{self.target_domain}.pth")
        checkpoint = torch.load(save_path)
        self.encoder.load_state_dict(checkpoint['encoder'])

        test_acc = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {test_acc:.2f}%")

        return test_acc