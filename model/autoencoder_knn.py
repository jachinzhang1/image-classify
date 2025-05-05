import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 x 16 x 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64 x 8 x 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128 x 4 x 4
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.upconv1(x)))  # 64 x 8 x 8
        x = F.relu(self.bn2(self.upconv2(x)))  # 32 x 16 x 16
        x = F.relu(self.bn3(self.upconv3(x)))  # 16 x 32 x 32
        x = torch.sigmoid(self.conv_out(x))     # 3 x 32 x 32
        return x


class AutoencoderKNN(nn.Module):
    def __init__(self, num_classes=10, n_neighbors=5, embedding_dim=128):
        super(AutoencoderKNN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # 分类头 - 用于标准训练流程
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # KNN classifier - 用于后续KNN方法
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.num_classes = num_classes
        self.fitted = False
        
    def encode(self, x):
        """编码器部分，提取特征嵌入"""
        features = self.encoder(x)
        embedding = self.embedding(features)
        return embedding, features
    
    def forward(self, x):
        """默认forward返回分类logits，兼容CrossEntropyLoss"""
        embedding, _ = self.encode(x)
        return self.classifier(embedding)  # 返回分类结果

    # 以下方法用于自编码器和KNN功能，不影响标准训练
    def get_reconstruction(self, x):
        """获取重建图像"""
        _, features = self.encode(x)
        return self.decoder(features)
        
    def fit_knn(self, dataloader, device='cuda'):
        """训练KNN分类器"""
        self.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                embedding, _ = self.encode(images)
                all_embeddings.append(embedding.cpu().numpy())
                all_labels.append(labels.numpy())
        
        # 拼接批次数据
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        # 训练KNN分类器
        self.knn.fit(embeddings, labels)
        self.fitted = True
        return self
    
    def predict_knn(self, x, return_proba=False):
        """使用KNN进行预测"""
        if not self.fitted:
            raise RuntimeError("KNN classifier has not been fitted yet. Call fit_knn first.")
        
        self.eval()
        with torch.no_grad():
            embedding, _ = self.encode(x)
            embedding = embedding.cpu().numpy()
        
        if return_proba:
            return self.knn.predict_proba(embedding)
        else:
            return self.knn.predict(embedding)
    
    def train_autoencoder(self, dataloader, optimizer=None, epochs=10, device='cuda'):
        """独立训练自编码器部分"""
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        criterion = nn.MSELoss()
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for images, _ in dataloader:
                images = images.to(device)
                
                # 前向传播
                _, features = self.encode(images)
                reconstructed = self.decoder(features)
                
                # 计算重建损失
                loss = criterion(reconstructed, images)
                
                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {avg_loss:.4f}')


