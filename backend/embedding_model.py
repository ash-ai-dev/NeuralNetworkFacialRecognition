import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> (128, 8, 8)
        )

        self.fc = nn.Linear(128 * 8 * 8, embedding_dim)

    def forward(self, x):
        x = self.conv_layers(x)  # shape: (batch_size, 128, 8, 8)
        x = x.reshape(x.size(0), -1)  # flatten
        embedding = self.fc(x)     # shape: (batch_size, embedding_dim)
        embedding = F.normalize(embedding, p=2, dim=1)  # optional: L2 normalize
        return embedding
