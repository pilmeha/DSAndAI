import torch
import math
from typing import Optional
from copy import deepcopy

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class Embedding(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int, pad_index: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_index)
        self.pos_encoding = PositionalEncoding(d_model, 5000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        return x
    
#flash-attention v2
class SDPA(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        factor = math.sqrt(self.d_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) / factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn_weights = torch.nn.Softmax(dim=-1)(scores)
        out = torch.matmul(attn_weights, v)
        return out

class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # self.dropout = dropout

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        self.attention = SDPA(d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bs = q.size(0)

        q = self.q_proj(q).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if mask is not None:
            mask = mask.unsqueeze(1)

        x = self.attention(q, k, v, mask)

        x = x.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.d_model)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class EncoderLayer(torch.nn.Module):
    def __init__(self, mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.attention = deepcopy(mha)
        self.ffn = deepcopy(ffn)
        self.layernorm1 = torch.nn.LayerNorm(mha.d_model)
        self.layernorm2 = torch.nn.LayerNorm(mha.d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x_norm = self.layernorm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm, mask)
        x_norm = self.layernorm2(x)
        x = self.dropout(x + self.ffn(x_norm))
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, mha: MultiheadAttention, cross_mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.attention = deepcopy(mha)
        self.cross_attention = deepcopy(cross_mha)
        self.ffn = deepcopy(ffn)
        self.layernorm1 = torch.nn.LayerNorm(mha.d_model)
        self.layernorm2 = torch.nn.LayerNorm(mha.d_model)
        self.layernorm3 = torch.nn.LayerNorm(mha.d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        x_norm = self.layernorm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm, tgt_mask)
        x_norm = self.layernorm2(x)
        x = x + self.cross_attention(x_norm, encoder_memory, encoder_memory, src_mask)
        x_norm = self.layernorm3(x)
        x = self.dropout(x + self.ffn(x_norm))
        return x

class Encoder(torch.nn.Module):
    def __inti__(self, encoder_layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, encoder_memory, src_mask, tgt_mask)
        return x