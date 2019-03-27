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
        
        self.conv1 = nn.Conv2d(input_size, 15, 3, stride=1, padding=1).to(device1)
        self.conv2 = nn.Conv2d(15, 12, 3, stride=1, padding=1).to(device1)
        self.conv3 = nn.Conv2d(12, 12, 3, stride=1, padding=1).to(device1)
        
        
    def forward(self, adjacency, atoms, membership):
        
        start = time.time()
        
        x_in = self.embedding(atoms).to(self.device0)
        
        hidden_states = []
        
        for i in range(self.n_conv):
            x_in = self.ggn(adjacency, x_in)
            
        h_final = torch.mm(membership, x_in)
        
        h_final = h_final.unsqueeze(1)
        
        orig_dim = h_final.shape
        
        h_final = (torch.add(h_final, h_final.transpose(0, 1)) / 2).view(1, self.input_size, int(orig_dim[0]), int(orig_dim[0]))
        
        end = time.time()
#         print('\nGraph conv: {}'.format(end - start))
        start = time.time()
        
        h_final = h_final.to(self.device1)
        
        out = self.conv1(h_final)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out).squeeze()
        
        out = torch.sigmoid(torch.add(out, out.transpose(1, 2)) / 2)
        
        end = time.time()
#         print('2D conv: {}'.format(end - start))
        
        return out
    