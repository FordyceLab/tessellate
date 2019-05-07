import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GGNUnit(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size, input_size)
        
    def forward(self, x_in, h_prev):
        h_t = self.gru(x_in, h_prev)
        
        return h_t


class FFN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, 12)
        self.linear2 = nn.Linear(12, 12)
        
    def forward(self, input_vec):
        x = self.linear1(input_vec)
        x = F.relu(x)
        x = self.linear2(x)
        out = torch.sigmoid(x)
        
        return(out)
        
        
    
class Network(nn.Module):
    def __init__(self, input_size, n_atom_conv, n_res_conv, device0, device1):
        super().__init__()
        
        self.device0 = device0
        self.device1 = device1
        
        self.input_size = input_size
        
#         self.embedding = nn.Embedding(116,
#                                       input_size,
#                                       scale_grad_by_freq=True)
        
        embeddings = torch.randn(116, input_size)
        
        for i in range(len(embeddings)):
            embeddings[i, 0] = i + 1
        
        self.embedding = nn.Embedding.from_pretrained(embeddings,
                                                      freeze=False,
                                                      scale_grad_by_freq=True)
        
        self.n_atom_conv = n_atom_conv
        self.n_res_conv = n_res_conv
        
        self.atom_linear = [nn.Linear(input_size, input_size, bias=False).to(device0) for layer in range(self.n_atom_conv)]
        self.res_linear = [nn.Linear(input_size, input_size, bias=False).to(device0) for layer in range(self.n_res_conv)]
        
#         self.atom_ggn = GGNUnit(input_size).to(device0)
#         self.res_ggn = GGNUnit(input_size).to(device0)
        
        self.ffn = FFN(input_size).to(device0)
        
        
    def forward(self, atom_adjacency, res_adjacency, atoms, membership, combos):
        
        x_in = self.embedding(atoms).to(self.device0)
        
        for i in range(self.n_atom_conv):
            x_in = F.relu(self.atom_linear[i](atom_adjacency.mm(x_in)))
            
        x_in = torch.mm(membership, x_in)
        
        for i in range(self.n_res_conv):
            x_in = F.relu(self.res_linear[i](res_adjacency.mm(x_in)))
#             x_in = x_out
#             x_out = self.res_ggn(x_in_prop, x_in)
            
#         x_in = x_in.to(self.device0)
        
        mean_combos = torch.mm(combos, x_in)
        
        out = self.ffn(mean_combos)
        
        return out
    