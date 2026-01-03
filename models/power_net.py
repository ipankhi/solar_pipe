import torch
import torch.nn as nn

class PowerNet(nn.Module):
    def __init__(self, input_dim, hidden=512, drop=0.18):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.act2 = nn.SiLU()
        self.residual = nn.Linear(input_dim, hidden)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1)
        )

    def forward(self, x):
        r = self.residual(x)
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.act2(self.ln2(self.fc2(h)))
        h = self.drop(h + r)
        return self.head(h)
