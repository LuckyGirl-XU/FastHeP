import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_size, output_size):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, output_size)

    def forward(self, x):
        
        Q = self.queries(x)  
        K = self.keys(x)     
        V = self.values(x)   

        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.embed_size ** 0.5
        attention_weights = F.softmax(attention_scores, dim=-1)

        
        out = torch.matmul(attention_weights, V)  
        return out
