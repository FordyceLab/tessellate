import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import time


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

        self.linear1 = nn.Linear(input_size, 25)
        self.linear2 = nn.Linear(25, 12)
        
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
                                      scale_grad_by_freq=True)
        
        self.n_conv = n_conv
        
        self.ggn = GGNUnit(input_size).to(device0)
        
        self.ffn = FFN(25).to(device1)
        
        self.conv1 = nn.Conv2d(input_size, 4, 3, stride=1, padding=1).to(device1)
        self.conv2 = nn.Conv2d(4, 12, 3, stride=1, padding=1).to(device1)
        
        
    def forward(self, adjacency, atoms, membership):
        
        x_in = self.embedding(atoms).to(self.device0)
        
        hidden_states = []
        
        for i in range(self.n_conv):
            x_in = self.ggn(adjacency, x_in)
            
        h_final = torch.mm(membership, x_in)
        
        #h_final = h_final.unsqueeze(1)
        
        #orig_dim = h_final.shape
        
        #h_final = (torch.add(h_final, h_final.transpose(0, 1)) / 2).view(1, self.input_size, int(orig_dim[0]), int(orig_dim[0]))
        
        h_final = h_final.to(self.device1)
        
        
        preds = []
        
        for i in range(len(h_final) - 1):
            for j in range(i, len(h_final)):
                mean_vec = torch.add(h_final[i], h_final[j]) / 2
                preds.append(self.ffn(mean_vec))
                
        mean_vec = torch.add(h_final[j], h_final[j]) / 2    
        preds.append(self.ffn(mean_vec))
        out = torch.cat(preds, 0)
        
#         out = self.conv1(h_final)
#         out = F.relu(out)
#         out = self.conv2(out).squeeze()
#         
#         out = torch.sigmoid(torch.add(out, out.transpose(1, 2)) / 2)
        
        return out
    