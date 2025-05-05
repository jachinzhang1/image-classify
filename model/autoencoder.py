import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [B, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # Output: [B, 32, 32, 32]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # Output: [B, 64, 16, 16]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Output: [B, 128, 8, 8]
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        # Output: [B, 128, 4, 4]
        self.flatten = nn.Flatten()
        # Output: [B, 128*4*4]
        self.fc = nn.Linear(128 * 4 * 4, 512)
        # Output: [B, 512]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [B, 512]
        self.fc = nn.Linear(512, 128 * 4 * 4)
        # Output: [B, 128*4*4]
        self.unflatten = nn.Unflatten(1, (128, 4, 4))
        # Output: [B, 128, 4, 4]
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        # Output: [B, 64, 8, 8]
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1)
        # Output: [B, 32, 16, 16]
        self.deconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1)
        # Output: [B, 16, 32, 32]
        self.deconv4 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        # Output: [B, 3, 32, 32]

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def encoder_decoder(self, x):
        # Autoencoder path - returns reconstructed image
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def forward(self, x):
        # Classification path - returns class logits
        encoded = self.encoder(x)
        return self.classifier(encoded)
    
    def train_autoencoder(
        self, 
        dataloader, 
        optimizer, 
        epochs, 
        device, 
        progress_callback=None,
        total_iterations=None,
        controller=None
    ):
        """Pre-train the autoencoder part"""
        self.train()
        
        # Use MSE loss for image reconstruction
        criterion = nn.MSELoss()
        
        current_iter = 0
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                
                # Check if training should be stopped
                if controller and controller.should_stop:
                    print("Autoencoder pre-training cancelled")
                    return
                
                # Forward pass
                reconstructed = self.encoder_decoder(data)
                loss = criterion(reconstructed, data)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                current_iter += 1
                
                # Update progress
                if progress_callback and total_iterations:
                    progress_callback.emit(current_iter, total_iterations, -epoch-1)  # Negative epoch to indicate pre-training
            
            # Print average loss after each epoch
            avg_loss = total_loss / batch_count
            print(f"Autoencoder Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
        
        print(f"Autoencoder pre-training completed after {epochs} epochs")


