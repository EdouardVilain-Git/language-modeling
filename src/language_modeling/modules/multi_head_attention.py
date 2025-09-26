import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Dimensions
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_k = config["d_k"]
        self.d_v = config["d_v"]

        # Query, Key, Value and final projection matrices
        self.Wq = nn.Linear(in_features=self.d_model, out_features=self.n_heads * self.d_k, bias=False)
        self.Wk = nn.Linear(in_features=self.d_model, out_features=self.n_heads * self.d_k, bias=False)
        self.Wv = nn.Linear(in_features=self.d_model, out_features=self.n_heads * self.d_v, bias=False)
        self.Wo = nn.Linear(in_features=self.n_heads * self.d_v, out_features=self.d_model, bias=False)

        # Boolean autoregressive flag
        self.is_autoregressive = config["is_autoregressive"]

        # Dropout
        self.attn_dropout = config["attn_dropout"]
        self.dropout = nn.Dropout(p=self.attn_dropout)

    def forward(self, x):
        # Compute Q, K, V matrics
        Q = self.Wq(x) # [batch, seq length, heads * d_k]
        K = self.Wk(x) # [batch, seq length, heads * d_k]
        V = self.Wv(x) # [batch, seq length, heads * d_v]

        # Reshape into [batch, heads, seq length, dim]
        B, T, _ = x.shape
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_v).transpose(1, 2)

        # Compute attention logits
        attn_logits = Q @ K.transpose(-1, -2) * (self.d_k ** -0.5) # [batch, heads, seq length, seq length]

        # Apply autoregressive mask if needed
        if self.is_autoregressive:
            autoregressive_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_logits = attn_logits.masked_fill(autoregressive_mask, float("-inf"))

        # Compute scores
        weights = torch.softmax(attn_logits, dim=-1)

        # Apply dropout
        weights = self.dropout(weights)

        # Multiply with value
        attn_scores = weights @ V

        # Concatenate all attention heads together
        attn_scores = attn_scores.transpose(1,2).contiguous().view(B, T, self.n_heads * self.d_v)

        # Project into the final dimension
        return self.Wo(attn_scores)
