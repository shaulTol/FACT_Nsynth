# Updated DeepAllTrainer with proper validation-test split
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from audio_resnet import AudioResNet18
from data_loaders import get_split_dataloaders, get_eval_dataloader, download_nsynth_dataset
from deep_all_resnet import DeepAllResNet18

class DeepAllTrainer:
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
        
        # Initialize the model
        self.model = DeepAllResNet18(num_classes=num_classes).to(device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets, _) in enumerate(tqdm(self.train_loader)):
            # Move to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total
        
        return train_loss, train_acc
    
    def evaluate(self, dataloader):
        """Evaluate the model on the given dataloader."""
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs, _ = self.model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100.0 * correct / total
    
    def train(self, epochs=50, save_dir='./deepall_models'):
        """Train for specified number of epochs."""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training on source domains: {self.source_domains}")
        print(f"Target domain for evaluation: {self.target_domain}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_acc = self.evaluate(self.val_loader)
            
            print(f"Epoch: {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = os.path.join(save_dir, f"deepall_model_{'-'.join(self.source_domains)}_to_{self.target_domain}.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                }, save_path)
                print(f"Model saved to {save_path}")
            
            print("-" * 50)
        
        # Test with the best model
        save_path = os.path.join(save_dir, f"deepall_model_{'-'.join(self.source_domains)}_to_{self.target_domain}.pth")
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_acc = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        return test_acc