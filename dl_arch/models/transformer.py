"""
Transformer components and layers
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x (seq_len, batch_size, d_model) - Input embeddings tensor with sequence length,
               batch size, and model dimension
        Output: x (seq_len, batch_size, d_model) - Input embeddings with positional encoding added

        Purpose: Add sinusoidal positional encoding to input embeddings to provide sequence
                position information to the transformer model.

        Mathematical formula:
            PE(pos,2i) = sin(pos/10000^(2i/d_model))
            PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
            output = x + PE[:seq_len, :]

        Tensor flow:
            x: (L, B, D) + PE: (L, D) -> output: (L, B, D)
        Where L=seq_len, B=batch_size, D=d_model
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Input: Q (batch_size, num_heads, seq_len, d_k) - Query tensor for each attention head
               K (batch_size, num_heads, seq_len, d_k) - Key tensor for each attention head  
               V (batch_size, num_heads, seq_len, d_k) - Value tensor for each attention head
               mask (batch_size, seq_len) or (batch_size, seq_len, seq_len) - Optional attention mask
        Output: output (batch_size, num_heads, seq_len, d_k) - Attention-weighted values
                attention_weights (batch_size, num_heads, seq_len, seq_len) - Attention scores

        Purpose: Compute scaled dot-product attention mechanism, the core operation of transformers.
                Calculates attention weights between queries and keys, then applies to values.

        Mathematical formula:
            scores = QK^T / √d_k
            attention_weights = softmax(scores + mask)
            output = attention_weights × V

        Tensor flow:
            Q: (B, H, L, D) × K^T: (B, H, D, L) -> scores: (B, H, L, L)
            scores -> softmax -> weights: (B, H, L, L) × V: (B, H, L, D) -> output: (B, H, L, D)
        Where B=batch_size, H=num_heads, L=seq_len, D=d_k
        """

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Expand mask to match scores dimensions
            # scores: (batch_size, num_heads, seq_len, seq_len)
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            if mask.dim() == 2:  # (batch_size, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(
                    2)  # (batch_size, 1, 1, seq_len)
            elif mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Input: query (batch_size, seq_len, d_model) - Query sequences
               key (batch_size, seq_len, d_model) - Key sequences  
               value (batch_size, seq_len, d_model) - Value sequences
               mask (batch_size, seq_len) - Optional attention mask
        Output: output (batch_size, seq_len, d_model) - Multi-head attention output

        Purpose: Perform multi-head attention by projecting inputs to multiple subspaces,
                computing attention in parallel, and combining results.

        Mathematical formula:
            Q = W_Q × query, K = W_K × key, V = W_V × value
            MultiHead(Q,K,V) = Concat(head₁, ..., head_h) × W_O
            where head_i = Attention(Q_i, K_i, V_i)

        Tensor flow:
            query,key,value: (B, L, D) -> Q,K,V: (B, L, D) -> heads: (B, H, L, D/H)
            -> attention: (B, H, L, D/H) -> concat: (B, L, D) -> output: (B, L, D)
        Where B=batch_size, L=seq_len, D=d_model, H=num_heads
        """
        batch_size = query.size(0)

        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1,
                                 self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1,
                               self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1,
                                 self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear transformation
        output = self.w_o(attention_output)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Input: x (batch_size, seq_len, d_model) - Input sequence representations
               mask (batch_size, seq_len) - Optional attention mask for padding tokens
        Output: x (batch_size, seq_len, d_model) - Transformed sequence representations

        Purpose: Apply one transformer encoder layer consisting of multi-head self-attention
                followed by position-wise feed-forward network, both with residual connections
                and layer normalization.

        Mathematical formula:
            attn_out = SelfAttention(x, x, x, mask)
            x₁ = LayerNorm(x + Dropout(attn_out))
            ff_out = FeedForward(x₁)
            x₂ = LayerNorm(x₁ + Dropout(ff_out))

        Tensor flow:
            x: (B, L, D) -> self_attention -> (B, L, D) -> residual+norm -> (B, L, D)
            -> feed_forward -> (B, L, D) -> residual+norm -> (B, L, D)
        Where B=batch_size, L=seq_len, D=d_model
        """
        # Self-attention with residual connection
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Input: x (batch_size, tgt_len, d_model) - Target sequence representations
               memory (batch_size, src_len, d_model) - Encoder output (source representations)
               tgt_mask (batch_size, tgt_len) - Mask for target sequence (causal + padding)
               memory_mask (batch_size, src_len) - Mask for source sequence (padding only)
        Output: x (batch_size, tgt_len, d_model) - Transformed target representations

        Purpose: Apply one transformer decoder layer with masked self-attention on targets,
                cross-attention between targets and source, and feed-forward network.
                All sub-layers have residual connections and layer normalization.

        Mathematical formula:
            self_attn_out = MaskedSelfAttention(x, x, x, tgt_mask)
            x₁ = LayerNorm(x + Dropout(self_attn_out))
            cross_attn_out = CrossAttention(x₁, memory, memory, memory_mask)
            x₂ = LayerNorm(x₁ + Dropout(cross_attn_out))
            ff_out = FeedForward(x₂)
            x₃ = LayerNorm(x₂ + Dropout(ff_out))

        Tensor flow:
            x: (B, T, D) -> self_attention -> (B, T, D) -> cross_attention with memory: (B, S, D)
            -> (B, T, D) -> feed_forward -> (B, T, D)
        Where B=batch_size, T=tgt_len, S=src_len, D=d_model
        """
        # Self-attention with residual connection
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-attention with residual connection
        cross_attn_output = self.cross_attention(
            x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
