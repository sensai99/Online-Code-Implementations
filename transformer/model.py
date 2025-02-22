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
        # math.sqrt?
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
