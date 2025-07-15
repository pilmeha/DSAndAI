import torch

from tokenizers import Tokenizer
from typing import Tuple

from layers import Embedding, MultiheadAttention, FeedForward, EncoderLayer, DecoderLayer, Encoder, Decoder

def get_pad_mask(x, pad_token):
    return (x != pad_token).unsqueeze(-2)

def get_subsequent_mask(x):
    bs, s, = x.size()
    mask = torch.tril(torch.ones(s, s)).bool()
    mask = mask.unsqueeze(0).expand(bs, -1, -1)
    return mask


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 6,
        vocab_size: int = 1,
        pad_index: int = 1,
        dropout: float = 0.1,
        max_len: int = 64,
        tokenizer: Tokenizer = None,
        device: str = 'cuda',
    ):
        
        super().__init__()
        self.d_model = d_model
        self.num_head = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.dropout = dropout
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device

        self.src_embedding = Embedding(d_model, vocab_size, pad_index)
        self.tgt_embedding = Embedding(d_model, vocab_size, pad_index)
        mha = MultiheadAttention(d_model, num_heads, dropout)
        cross_mha = MultiheadAttention(d_model, num_heads, dropout)
        ffn = FeedForward(d_model, d_ff, dropout)
        enc_layer = EncoderLayer(mha, ffn, dropout)
        dec_layer = DecoderLayer(mha, cross_mha, ffn, dropout)
        self.encoder = Encoder(enc_layer, num_layers)
        self.decoder = Decoder(dec_layer, num_layers)
        self.normalize = torch.nn.LayerNorm(d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)

    def predict(self, x: torch.Tensor):
        memory, mask = self.encode_src(x)
        bos_token = self.tokenizer.token_to_id('<s>')
        eos_token = self.tokenizer.token_to_id('</s>')
        pad_token = self.tokenizer.token_to_id('<pad>')

        batch_size = memory.size(0)

        tgt_tensor = torch.full(
            (batch_size, self.max_len),
            pad_token, 
            device=self.device,
            dtype=torch.long
        )
        tgt_tensor[:, 0] = bos_token
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for step in range(1, self.max_len):
            if finished.all():
                break
            active_indices = torch.nonzero(~finished).view(-1)
            active_memory = memory[active_indices]
            active_tgt = tgt_tensor[active_indices]
            active_mask = mask[active_indices]
            out = self.decode_tgt(active_tgt, active_memory, active_mask)
            next_token = out[:, -1, :].argmax(dim=-1)
            tgt_tensor[active_indices, step] = next_token

            finished[active_indices] |= (next_token == eos_token)
        return tgt_tensor


    def encode_src(self, x):
        pad_mask = get_pad_mask(x, self.pad_index)
        self.src_embedding(x)
        return self.encoder(x, pad_mask), pad_mask
    
    def decode_tgt(self, x, memory, src_mask):
        tgt_mask = get_pad_mask(x, self.pad_index) & get_subsequent_mask(x).to(self.device)
        x = self.tgt_embedding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        x = self.normalize(x)
        return self.linear(x)
    
    def forward(self, x, y):
        memory, mask = self.encode_src(x)
        out = self.decode_tgt(y, memory, mask)
        return out
        
