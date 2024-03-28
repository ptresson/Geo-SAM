import torch.nn as nn

class SimpleFCHead(nn.Module):
    def __init__(self, embed_dim, nb_cls):
        super().__init__()
        self.intermediate_dim = 512
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embed_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, nb_cls),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        cls = self.linear_relu_stack(x)
        return cls