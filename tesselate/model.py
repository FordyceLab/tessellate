import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GGNUnit(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.linear = nn.Linear(input_size, input_size, bias=True)
        self.gru = nn.GRUCell(input_size, input_size)
        
    def forward(self, adjacency, h_prev):
        x_in = self.linear(adjacency.mm(h_prev))
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
    def __init__(self, input_size, n_conv, device0, device1):
        super().__init__()
        
        self.device0 = device0
        self.device1 = device1
        
        self.input_size = input_size
        
        self.embedding = nn.Embedding(116,
                                      input_size,
                                      scale_grad_by_freq=False)
        
        self.n_atom_conv = n_atom_conv
        self.n_atom_conv = n_atom_conv
        
        self.ggn = GGNUnit(input_size).to(device0)
        
        
        self.ffn = FFN(input_size).to(device1)
        
        
    def forward(self, adjacency, atoms, membership, combos):
        
        x_in = self.embedding(atoms).to(self.device0)
        
        hidden_states = []
        
        for i in range(self.n_conv):
            x_in = self.ggn_atom(atom_adjacency, x_in)
            
        x_in = torch.mm(membership, x_in).to(self.device1)
        
        for i in range(self.n_conv):
            x_in = self.ggn_res(res_adjacency, x_in)
        
        mean_combos = torch.mm(combos, x_in)
        
        out = self.ffn(mean_combos)
        
        return out
    