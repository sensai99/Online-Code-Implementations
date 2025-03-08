import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Create the embedding given an ID
        # Need to get the same embedding always for a given ID
        self.embedding = nn.Embedding(vocab_size, d_model)
        return

    def forward(self, x):
        # Diff: math.sqrt?
        # Why do we multiply by d_model
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # (1, seq_len, d_model)
        pe = torch.zeros((1, self.seq_len, self.d_model))

        # (seq_len, 1)
        pos = torch.arange(0, self.seq_len, dtype = torch.float).unsqueeze(1)
        den = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000) / self.d_model))

        temp = pos * den
        pe[0, :, 0::2] = torch.sin(temp)
        pe[0, :, 1::2] = torch.cos(temp)

        self.register_buffer('pe', pe)
        return
    
    def forward(self, x):
        # Not computing gradients as it's not necessary
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        
        # Why are we applying droput here?
        return self.dropout(x)
    
class LayerNorm(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super().__init__()
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
        std = torch.std(x, dim = -1, keepdim = True)
        return (self.alpha * (x - mu)/(std + self.eps)) + self.beta

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
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        
        # W_o weight matrix (applied at the end of the attention mechanism)
        self.w_o = nn.Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(dropout)
        return
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Then we need to apply the mask so that information from future tokens is not passed
            attn_scores.masked_fill_(mask == 0, -1e9)

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

        # Split the query, key and values to h equal parts (Multi-head)
        # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        # Transpose is done so that each head is pointing to the entire sequence - easy for matrix manipulation
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Attention
        attention_heads, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attention_heads = attention_heads.transpose(1, 2).contiguous().view(attention_heads.shape[0], -1, self.h * self.d_k)

        attention_heads = self.w_o(attention_heads)
        return attention_heads
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

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
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        # Pass through multiple EncoderBlocks
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
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()
        return

    def forward(self, x, decoder_mask, y, encoder_mask):
        # Pass through multiple DecoderBlocks
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
        # No need to apply softmax because the nn.CrossEntropyLoss already does it
        return self.proj(x)
    
     

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, src_emb: InputEmbeddings, src_pos: PositionalEmbeddings, decoder: Decoder, tgt_emb: InputEmbeddings, tgt_pos: PositionalEmbeddings, projectionLayer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.src_emb = src_emb
        self.src_pos = src_pos

        self.decoder = decoder
        self.tgt_emb = tgt_emb
        self.tgt_pos = tgt_pos
        
        self.projectionLayer = projectionLayer
        return

    def encode(self, src, src_mask):
        src = self.src_emb(src)

        # Adds positional emebddings to the input embeddings
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, tgt_mask, src, src_mask):
        # src - output of the encoder, will be passed to all decoder blocks
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, tgt_mask, src, src_mask)
    
    def project(self, x):
        return self.projectionLayer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, ff_hidden_dim: int = 2048) -> Transformer:
    '''
        @src_vocab_size: Vocabulary size of source sentence
        @tgt_vocab_size: Vocabulary size of target sentence and it can be different from source
        @d_model: Embedding dimension for a given token. It should be same across src and target sentence (i.e encoder and decder)
        N: Number of Encoder/Decoder blocks
        h: Number of attention heads in a block
        ff_hidden_dim: Dimension of the feed forward block's hidden layer 
    '''

    # Build input and positional embedding objects for src and target
    src_emb = InputEmbeddings(d_model, src_vocab_size)
    src_pos = PositionalEmbeddings(d_model, src_seq_len, dropout)

    tgt_emb = InputEmbeddings(d_model, tgt_vocab_size)
    tgt_pos = PositionalEmbeddings(d_model, tgt_seq_len, dropout)

    # Build the encoder block list
    # -> MultiHeadAttn block + FeedForward block
    encoder_block_layers = []
    for _ in range(N):
        encoder_block_layers.append(EncoderBlock(MultiHeadAttentionBlock(d_model, h, dropout), FeedForwardBlock(in_dim = d_model, hidden_dim = ff_hidden_dim, dropout = dropout), dropout))
    encoder_block_layers = nn.ModuleList(encoder_block_layers)

    # Encoder
    encoder = Encoder(encoder_block_layers)

    # Build the decoder block list
    # -> MultiHeadAttn block (self) + MultiHeadAttn block (cross) + FF block
    decoder_block_layers = []
    for _ in range(N):
        decoder_block_layers.append(DecoderBlock(MultiHeadAttentionBlock(d_model, h, dropout), MultiHeadAttentionBlock(d_model, h, dropout), FeedForwardBlock(in_dim = d_model, hidden_dim = ff_hidden_dim, dropout = dropout), dropout))
    decoder_block_layers = nn.ModuleList(decoder_block_layers)

    # Decoder
    decoder = Decoder(decoder_block_layers)

    # Create a projection layer instance
    # vocab_size is of target: Since we are predicting the words in target, we need the probability dist over target vocab
    projection_layer = ProjectionLayer(d_model, vocab_size = tgt_vocab_size)

    # Create a transformer instance
    transformer = Transformer(encoder, src_emb, src_pos, decoder, tgt_emb, tgt_pos, projection_layer)

    # Intialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


