import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config