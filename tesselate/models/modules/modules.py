import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


####################
# Embedding layers #
####################

class AtomEmbed(nn.Module):
    """
    Embed the atoms to fixed-length input vectors.
    
    Args:
    - num_features (int) - Size of the returned embedding vectors.
    - scale_grad_by_freq (bool) - Scale gradients by the inverse of
        frequency (default=True).
    """
    
    def __init__(self, n_features, scale_grad_by_freq=True):
        super(AtomEmbed, self).__init__()
        self.embedding = nn.Embedding(118,
                                      n_features,
                                      scale_grad_by_freq=scale_grad_by_freq)
        
    def forward(self, atomic_numbers):
        """
        Return the embeddings for each atom in the graph.
        
        Args:
        - atoms (torch.LongTensor) - Tensor (n_atoms) containing atomic numbers.
        
        Returns:
        - torch.FloatTensor of dimension (n_atoms, n_features) containing
            the embedding vectors.
        """
        
        # Get and return the embeddings for each atom
        embedded_atoms = self.embedding(atomic_numbers)
        return embedded_atoms


####################
# Attention layers #
####################

class GraphAttn(nn.Module):
    """
    Graph attention layer.
    
    Args:
    - in_features (int) - Number of input features.
    - out_features (int) - Number of output features.
    - dropout (bool) - P(keep) for dropout.
    - alpha (float) - Alpha value for leaky ReLU.
    """
    
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttn, self).__init__()
        
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
        - nodes (torch.FloatTensor) - Node feature matrix 
            (n_nodes, in_features).
        - adj (torch.FloatTensor) - Adjacency matrix (n_nodes, n_nodes).
        
        Returns:
        - torch.FloatTensor of dimension (n_nodes, n_nodes) of attentional
            coefficients where a_ij is the attention value of for node j with
            respect to node i.
        """
        
        # Node mat input shape = (n_nodes, out_features)
        node_fts = self.W(nodes)
        
        # Number of nodes in graph
        n_nodes = node_fts.size()[0]
        
        # Concatenate to generate input for attention mechanism
        # (n_nodes, n_nodes, 2*out_features)
        a_input = (torch.cat([node_fts.repeat(1, n_nodes).view(n_nodes *
                                                               n_nodes, -1),
                              node_fts.repeat(n_nodes, 1)], dim=1)
                   .view(n_nodes, n_nodes, 2 * self.out_features))
        
        # Get the attention logits
        # (n_nodes, n_nodes)
        e = self.activation(self.a(a_input).squeeze(2))
        
        # Create the mask based on adjacency matrix
        # Clip at 0.5
        mask = adj < 0.5
        e[mask] = float('-inf')
        
        # Get the attention coefficiencts
        # (n_nodes, n_nodes)
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        return attention


#####################
# Prediction layers #
#####################
    
class FCContactPred(nn.Module):
    """
    Fully connected layer to perform contact prediction.
    
    Args:
    - node_features (int) - Number of input features.
    - out_features (int) - Number of output prediction values.
    """
    
    def __init__(self, node_features, out_preds):
        super(FCContactPred, self).__init__()
        
        self.linear1 = nn.Linear(node_features, 25, bias=True)
        self.linear2 = nn.Linear(25, out_preds, bias=True)
    
    def forward(self, combined_nodes):
        """
        Predict pointwise multichannel contacts from summarized pairwise
        residue features.
        
        Args:
        - nodes (torch.FloatTensor) - Tensor of (convolved) node features
            (n_pairwise, n_features).

        Returns:
        - torch.FloatTensor (n_contacts, n_channels) containing the prediction
            for every potential contact point and every contact channel.
        """
        # Get the logits from the linear layer
        prelogits = self.linear1(combined_nodes)
        prelogits = F.leaky_relu(prelogits)
        logits = self.linear2(prelogits)
        
        # Apply sigmoid transform
        preds = torch.sigmoid(logits)
        
        return preds
    
    
class OrderInvContPred(nn.Module):
    """
    Order invariant contact prediction module.
    
    Args:
    - node_features (int) - Number of embedding features.
    - out_preds (int) - Number of output predictions (n_channels).
    """
    
    def __init__(self, node_features, out_preds):
        super(Condense, self).__init__()
        
        self.W1 = nn.Parameter(torch.randn((node_features*2, node_features*2),
                                           requires_grad=True))
        
        self.W2 = nn.Parameter(torch.randn((node_features*2, out_preds),
                                           requires_grad=True))
        
    def forward(self, concatenated_nodes):
        """
        Predict pointwise multichannel contacts from summarized pairwise
        residue features.
        
        Args:
        - concatenated_nodes (torch.FloatTensor) - Tensor of (convolved) node
            features (n_contacts, n_features * 2). All of the fully convolved
            features should be reversed along dim 1 for the second residue and
            then concatenated. This functionality can be obtained from the
            `cat_pairwise` function in the tesselate.models.functions module.

        Returns:
        - torch.FloatTensor (n_contacts, n_channels) containing the prediction
            for every potential contact point and every contact channel.
        """
        
        # Perform pointwise averaging to get
        W1_symm = self.W1 + self.W1.T
        W1_prime = (W1_symm + torch.rot90(self.W1, 2, (0, 1))) / 3
        W2_prime = (self.W2 + torch.flip(self.W2, dims=(0,))) / 2
        
        n_row = concatenated_nodes.shape[0]
        input_tens = concatenated_nodes.view((n_row, 1, -1))
        
        int_feats = concatenated_nodes.matmul(W1_prime)
        int_feats = F.relu(int_feats)
        
        logits = int_feats.matmul(W2_prime)
        
        preds = torch.sigmoid(logits).squeeze()
        
        return preds
