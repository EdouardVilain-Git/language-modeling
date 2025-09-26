import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Implements a Self-Attention layer.
    """

    def __init__(self, config):
        super().__init__()

        # Input, intermediary and output dims
        self.d_model = config["d_model"]
        self.d_k = config["d_k"]
        self.d_v = config["d_v"]

        # Is the self-attention 
        self.is_autoregressive = config["is_autoregressive"]

        # Attention dropout
        self.attn_dropout = config["attention_dropout"]
        self.dropout = nn.Dropout(p=self.attn_dropout)

        # Instantiate query, key, value, output proj
        self.Wq = nn.Linear(in_features=self.d_model, out_features=self.d_k, bias=False)
        self.Wk = nn.Linear(in_features=self.d_model, out_features=self.d_k, bias=False)
        self.Wv = nn.Linear(in_features=self.d_model, out_features=self.d_v, bias=False)
        self.Wo = nn.Linear(in_features=self.d_v, out_features=self.d_model, bias=False)


    def forward(self, x: torch.Tensor):
        # Compute Q, K, V matrices
        Q = self.Wq(x) # [batch, sequence length, d_k]
        K = self.Wk(x) # [batch, sequence length, d_k]
        V = self.Wv(x) # [batch, sequence length, d_v]

        # Compute attention scores
        attn_logits = Q @ K.transpose(-1, -2) * (self.d_k ** -0.5) # [batch, seq length, seq length]

        # Apply auto-regressive mask if needed
        if self.is_autoregressive:
            _, T, _ = attn_logits.shape
            autoreg_mask = torch.triu(torch.ones((T,T), device=x.device, dtype=torch.bool), diagonal=1)
            attn_logits = attn_logits.masked_fill(autoreg_mask, float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(attn_logits, dim=-1) # [batch, seq length, seq length]

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Multiply with value matrix and pass through output proj
        out = attn_weights @ V # [batch, seq length, d_v]
        out = self.Wo(out)

        return out