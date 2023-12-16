import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, num_features, feature_dim]
        scores = self.attention(x.mean(2))  # [batch_size, num_features, 1]
        return scores


class LearningModule(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LearningModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, num_features, feature_dim]
        x = x.mean(2)  # [batch_size, num_features]
        return self.model(x)


class AFS(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super(AFS, self).__init__()
        self.attention_module = AttentionModule(in_channels, hidden_dim)
        self.learning_module = LearningModule(in_channels, num_classes)

    def forward(self, x):
        # x: [batch_size, num_features, feature_dim]
        scores = self.attention_module(x)  # [batch_size, num_features, 1]
        x = x * scores  # [batch_size, num_features, feature_dim]
        logits = self.learning_module(x)  # [batch_size, num_classes]
        return logits
