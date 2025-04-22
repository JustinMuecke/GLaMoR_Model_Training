import torch.nn as nn
import torch
import torch.nn.functional as F

class WideMLP(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(WideMLP, self).__init__()
        self.fc1 = nn.Linear(100, 1024)  # Input to hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 2)    # Hidden to output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
