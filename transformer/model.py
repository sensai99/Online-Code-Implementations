import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Create the embedding given an ID
        # Need to get the same embedding always for a given ID
        self.embedding = nn.Embedding(vocab_size, d_model)
        return

    def forward(self, x):
        # Diff: math.sqrt?
        # Why do we multiply by d_model
        return self.embedding(x) * torch.sqrt(self.d_model)


class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # (1, seq_len, d_model)
        self.pe = torch.zeros((1, self.seq_len, self.d_model))

        # (seq_len, 1)
        pos = torch.arange(0, self.seq_len, dtype = torch.float).unsqueeze(1)
        den = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(10000.0) / self.d_model))

        temp = pos/den
        self.pe[0, :, 0::2] = torch.sin(temp)
        self.pe[0, :, 1::2] = torch.cos(temp)

        self.register_buffer('pe', self.pe)
        return
    
    def forward(self, x):
        # Not computing gradients as it's not necessary
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        
        # Why are we applying droput here?
        return self.dropout(x)
    
class LayerNorm(nn.Module):

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        
        # 1. What is the significance of alpha and beta?
        # Not making the dist to be (0, 1) - this might be too restrictive
        # These params will allow the model to make slight changes
        #
        # 2. Why parameter?
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        return
    
    def forward(self, x):
        mu = torch.mean(x, dim = -1, keepdim = True)
        var = torch.var(x, dim = -1, keepdim = True)
        return (self.alpha * (x - mu)/torch.sqrt(var + self.eps)) + self.beta

class FeedForwardBlock(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        
        # Here in_dim will be the d_model coming from the attention block
        # We transform it to a hidden representation and then again
        # transform back to d_model
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, in_dim)
        return
    
    def forward(self, x):
        # (batch_len, seq_len, d_model) -> (batch_len, seq_len, hidden_dim) -> (batch_len, seq_len, d_model)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # Make sure d_model can be equally divided into h parts
        assert d_model % h == 0, "Unexpected! d_model is not divisible by h"

        self.d_k = d_model // h

        # Key, Query, Value weight matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # W_o weight matrix (applied at the end of the attention mechanism)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Droput(dropout)
        return
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attn_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            # Then we need to apply the mask so that information from future tokens is not passed
            attn_scores.masked_fill_(mask == 0, torch.inf)

        # (batch_size, h, seq_len, seq_len)
        attn_scores = torch.softmax(attn_scores, dim = -1)

        if dropout is not None:
            attn_scores = dropout(attn_scores)
        
        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)
        attention_heads = attn_scores @ value
        
        return attention_heads, attn_scores
    
    def forward(self, q, k, v, mask):
        # For the query, key, value using the appropriate weight matrices
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the query, key and values to h equal parts (Multi- head)
        # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        # Transpose is done so that each head is pointing to the entire sequence - easy for matrix manipulation
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Attention
        attention_heads, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attention_heads = attention_heads.transpose(1, 2)
        # Diff: Need to use continuous
        attention_heads = attention_heads.view(attention_heads.shape[0], attention_heads.shape[1], self.h * self.d_k)

        attention_heads = self.w_o(attention_heads)
        return attention_heads
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # src_mask is used to mask the padding objects such that informaion is not passed from padding objects to other tokens in the sequence
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers = nn.ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        return

    def forward(self, x, decoder_mask, y, encoder_mask):
        # y is the output from the Encoder
        # Decoder mask is applied in the self attention block of decoder
        # Encoder mask is applied in the cross attention block of decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, decoder_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, y, y, encoder_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.norm = LayerNorm()
        return

    def forward(self, x, decoder_mask, y, encoder_mask):
        for layer in self.layers:
            x = layer(x, decoder_mask, y, encoder_mask)
        
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        return
    
    def forward(self, x):
        # (batch_size, seq_len, d_model) - (batch_size, seq_len, vocab_size)
        return torch.softmax(self.proj(x), dim = -1)