import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class AtomEmbed(nn.Module):
    """
    Embed the atoms to fixed-length input vectors.
    
    Args:
    - num_features (int) - Size of the returned embedding vectors.
    - scale_grad_by_freq (bool) - Scale gradients by the inverse of
        frequency (default=True).
    """
    
    def __init__(self, num_features, scale_grad_by_freq=True):
        super().__init__()
        self.embedding = nn.Embedding(118,
                                      embedding_size,
                                      scale_grad_by_freq=scale_grad_by_freq)
        
    def forward(self, atomic_numbers):
        """
        Return the embeddings for each atom in the graph.
        
        Args:
        - atoms (torch.LongTensor) - Tensor containing atomic numbers.
        
        Returns:
        - torch.FloatTensor of dimension (num_atoms, embed_size) containing
            the embedding vectors.
        """
        
        embedded_atoms = self.embedding(atoms)
        return embedded_atoms
    
    
class GraphAttn(nn.Module):
    """
    Graph attention layer.
    
    Args:
    - in_features (int) - 
    - out_features (int) - 
    - dropout (bool) - 
    - alpha (float) - 
    - concat_out (bool) - 
    """
    
    def __init__(self, in_features, out_features, dropout, alpha):
        
        # Parameters
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Operations
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.activation = nn.LeakyReLU(self.alpha)
        
    def forward(self, nodes, adj):
        """
        Perform forward pass through graph attention layer.
        
        Args:
        - nodes (torch.FloatTensor) - Node feature matrix (n_nodes, in_features).
        - adj (torch.FloatTensor) - Adjacency matrix (n_nodes, n_nodes).
        """
        
        # Node mat input shape = (num_nodes, out_features)
        node_fts = self.linear1(node)
        
        # Number of nodes in graph
        n_nodes = node_fts.size()[0]
        
        # Concatenate to generate input for attention mechanism
        # (n_nodes, n_nodes, 2*out_features)
        a_input = (torch.cat([node_fts.repeat(1, n_nodes).view(n_nodes * n_nodes, -1),
                              node_fts.repeat(n_nodes, 1)], dim=1)
                   .view(n_nodes, n_nodes, 2 * self.out_features))
        
        # Get the attention logits
        # (n_nodes, n_nodes)
        e = self.activation(self.a(a_input).squeeze(2))
        
        # Create the mask based on adjacency matrix
        # Clip at 0.5
        mask = torch.where(adj > 0.5, 1, -9e15)
        
        # Get the attention coefficiencts
        # (n_nodes, n_nodes)
        attention = F.softmax(e * mask, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Get the output
        # (n_nodes, out_features)
        output = self.activation(torch.matmul(attention, node_fts))
        
        return output
    
    
class FCContactPred(nn.Module):
    """
    Fully connected layer to perform contact prediction.
    
    Args:
    - in_features (int) - Number of input features.
    """
    
    def __init__(self, in_features, out_features):
        
        self.linear = nn.Linear(in_features, out_features, bias=True)
    
    def forward(self, combined_nodes):
        """
        Predict pointwise multichannel contacts
        
        Args:
        - nodes (torch.FloatTensor) - Tensor of (convolved) node features
            (n_nodes, n_features).
        - combine (callable) - Function to combine nodes according to
            upper triangle of interaction matrix.
        """
        # Get the logits from the linear layer
        logits = self.linear(combined_nodes)
        
        # Apply sigmoid transform
        preds = F.sigmoid(logits)
        
        return preds

    
class GGNUnit(nn.Module):
    def __init__(self, input_size):
        super(GGNUnit, self).__init__()
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
        
        self.embedding = nn.Embedding(116,
                                      input_size,
                                      scale_grad_by_freq=True)
        
#         embeddings = torch.randn(116, input_size)
        
#         for i in range(len(embeddings)):
#             embeddings[i, 0] = i + 1
        
#         self.embedding = nn.Embedding.from_pretrained(embeddings,
#                                                       freeze=False,
#                                                       scale_grad_by_freq=True)
        
        self.n_atom_conv = n_atom_conv
        self.n_res_conv = n_res_conv
        
        self.atom_linear = nn.ModuleList([nn.Linear(input_size * (layer + 1), input_size, bias=False).to(device0) for layer in range(self.n_atom_conv)])
        self.res_linear = nn.ModuleList([nn.Linear(input_size * (layer + 1), input_size, bias=False).to(device0) for layer in range(self.n_res_conv)])
        
#         self.atom_ggn = GGNUnit(input_size).to(device0)
#         self.res_ggn = GGNUnit(input_size).to(device0)
        
        self.ffn = FFN(input_size).to(device0)
        
        
    def forward(self, atom_adjacency, res_adjacency, atoms, membership, combos):
        
        x_in = self.embedding(atoms).to(self.device0)
        x_out = x_in
        
        for i in range(self.n_atom_conv):
            x_out = F.relu(self.atom_linear[i](atom_adjacency.mm(x_in)))
            x_in = torch.cat([x_out, x_in], 1)
            
        x_in = torch.mm(membership, x_out)
        
        for i in range(self.n_res_conv):
            x_out = F.relu(self.res_linear[i](res_adjacency.mm(x_in)))
            x_in = torch.cat([x_out, x_in], 1)
            
#             x_in = x_out
#             x_out = self.res_ggn(x_in_prop, x_in)
            
#         x_in = x_in.to(self.device0)
        
        mean_combos = torch.mm(combos, x_out)
        
        out = self.ffn(mean_combos)
        
        return out