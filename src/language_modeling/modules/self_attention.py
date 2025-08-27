import torch
import torch.nn as nn

import math
from typing import Optional

class SelfAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.d_model = config["embedding_dim"]
        self.d_k = config["intermediate_dim"]
        self.d_v = config["output_dim"]
        self.scale = 1 / math.sqrt(self.d_k)

        self.Wq = nn.Linear(self.d_model, self.d_k, bias=False)
        self.Wk = nn.Linear(self.d_model, self.d_k, bias=False)
        self.Wv = nn.Linear(self.d_model, self.d_v, bias=False)

        self.attn_dropout = nn.Dropout(config["attn_dropout"])

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # Compute Q,K,V matrices
        Q = self.Wq(x) # (batch_size, n, intermediate_dim)
        K = self.Wk(x) # (batch_size, n, intermediate_dim)
        V = self.Wv(x) # (batch_size, n, output_dim)

        # Compute Q.K^T
        scores = Q @ K.transpose(-2, -1) * self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # Compute scores
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Multiply scores with value
        return attn @ V
