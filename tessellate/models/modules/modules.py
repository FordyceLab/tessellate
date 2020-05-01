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


class AtomOneHotEmbed(nn.Module):
    """
    Create one-hot embeddings for atom identities.
    """
    def __init__(self):
        super(AtomOneHotEmbed, self).__init__()
        
        self.idx_map = {
            6: 0,
            7: 1,
            8: 2,
            15: 3,
            16: 4
        }
        
    def forward(self, atomic_numbers):
        """
        Return the embeddings for each atom in the graph.
        
        Args:
        - atoms (torch.LongTensor) - Tensor (n_atoms) containing atomic numbers.
        
        Returns:
        - torch.FloatTensor of dimension (n_atoms, n_features) containing
            the embedding vectors.
        """
        
        embedded_atoms = torch.zeros((len(atomic_numbers), 6))
        for i, j in enumerate(atomic_numbers):
            j = int(j)
            if j in self.idx_map:
                embedded_atoms[i, self.idx_map[j]] = 1
            else:
                embedded_atoms[i, 5] = 1
            
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
        self.W1 = nn.Linear(in_features, out_features, bias=True)
        self.W2 = nn.Linear(out_features, out_features, bias=True)
        self.W3 = nn.Linear(out_features, out_features, bias=True)
        self.W4 = nn.Linear(out_features, out_features, bias=True)
        
        
        self.a = nn.Linear(2*out_features, 1, bias=True)
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
        node_fts = F.relu(self.W1(nodes))
        node_fts = F.relu(self.W2(node_fts))
        node_fts = F.relu(self.W3(node_fts))
        node_fts = (self.W4(node_fts))
        
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
        
        self.f_attn = attention.detach().cpu()
        
        return self.activation(torch.matmul(attention, node_fts))


class MultiHeadGraphAttn(nn.Module):
    """
    Multi-head graph attention layer.
    
    Args:
    - n_head (int) - Number of heads for the attention layer.
    - in_features (int) - Number of total input features.
    - out_features (int) - Number of output features per head.
    - dropout (bool) - P(keep) for dropout.
    - alpha (float) - Alpha value for leaky ReLU.
    """
    
    def __init__(self, n_heads, in_features, out_features, dropout, alpha):
        super(MultiHeadGraphAttn, self).__init__()
        
        # Parameters
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Operations
        heads = [GraphAttn(in_features, out_features, dropout, alpha)
                 for i in range(n_heads)]
        self.attn_heads = nn.ModuleList(heads)
        
    def forward(self, nodes, adj):
        """
        Perform forward pass through multi-head graph attention layer.
        
        Args:
        - nodes (torch.FloatTensor) - Node feature matrix 
            (n_nodes, in_features).
        - adj (torch.FloatTensor) - Adjacency matrix (n_nodes, n_nodes).
        
        Returns:
        - torch.FloatTensor of dimension (n_nodes, n_nodes) of attentional
            coefficients where a_ij is the attention value of for node j with
            respect to node i.
        """
        
        return torch.cat([head(nodes, adj) for head in self.attn_heads],
                         dim = 1)
    

class CondenseAttn(nn.Module):
    """
    Graph attention layer.
    
    Args:
    - in_features (int) - Number of input features.
    - out_features (int) - Number of output features.
    - dropout (bool) - P(keep) for dropout.
    - alpha (float) - Alpha value for leaky ReLU.
    """
    
    def __init__(self, in_features, out_features, dropout, alpha):
        super(CondenseAttn, self).__init__()
        
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
        logits = self.activation(self.a(a_input).squeeze(2))
        logits = torch.matmul(adj, logits)
        
        # Create the mask based on adjacency matrix
        # Clip at 0.5
        mask = adj < 0.5
        logits[mask] = float('-inf')
        
        # Get the attention coefficiencts
        # (n_nodes, n_nodes)
        attention = F.softmax(logits, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        self.f_attn = attention.detach().cpu()
        
        return self.activation(torch.matmul(attention, node_fts))


#########################
# Position-aware layers #
#########################

# # PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=False):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


#####################
# Prediction layers #
#####################

class AttnPred(nn.Module):
    """
    Graph attention layer.
    
    Args:
    - in_features (int) - Number of input features.
    - out_features (int) - Number of output features.
    - dropout (bool) - P(keep) for dropout.
    - alpha (float) - Alpha value for leaky ReLU.
    """
    
    def __init__(self, in_features, out_features, dropout, alpha):
        super(PredAttn, self).__init__()
        
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
        logits = self.activation(self.a(a_input).squeeze(2))
        logits = torch.matmul(adj, logits)
        
        # Create the mask based on adjacency matrix
        # Clip at 0.5
        mask = adj < 0.5
        logits[mask] = float('-inf')
        
        # Get the attention coefficiencts
        # (n_nodes, n_nodes)
        attention = F.softmax(logits, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        self.f_attn = attention.detach().cpu()
        
        return self.activation(torch.matmul(attention, node_fts))


class FCContactPred(nn.Module):
    """
    Fully connected layer to perform contact prediction.
    
    Args:
    - node_features (int) - Number of input features.
    - out_features (int) - Number of output prediction values.
    """
    
    def __init__(self, node_features, out_preds, layers=3):
        super(FCContactPred, self).__init__()
        
        self.linear_first = nn.Linear(node_features, 25, bias=True)
        
        self.int_layers = nn.ModuleList([nn.Linear(25, 25, bias=True)
                                         for i in range(layers - 2)])

        self.linear_final = nn.Linear(25, out_preds, bias=True)
    
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
        prelogits = self.linear_first(combined_nodes)
        prelogits = F.relu(prelogits)
        
        for layer in self.int_layers:
            prelogits = layer(prelogits)
            prelogits = F.relu(prelogits)

        logits = self.linear_final(prelogits)
        
        return logits
    
    
class OrderInvContPred(nn.Module):
    """
    Order invariant contact prediction module.
    
    Args:
    - node_features (int) - Number of embedding features.
    - out_preds (int) - Number of output predictions (n_channels).
    """
    
    def __init__(self, node_features, out_preds, layers=3):
        super(OrderInvContPred, self).__init__()
        
        self.weight_mats = nn.ParameterList([])
        
        for i in range(layers):
            self.weight_mats.append(nn.Parameter(torch.randn((node_features*2,
                                                              node_features*2),
                                                 requires_grad=True)))
        
        self.W_out = nn.Parameter(torch.randn((node_features*2, out_preds),
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
        
        int_feats = concatenated_nodes
        
        for param in self.weight_mats:
            int_feats = concatenated_nodes.matmul(self.make_symmetric(param))
            int_feats = F.relu(int_feats)
        
        
        W_out_prime = (self.W_out + torch.flip(self.W_out, dims=(0,))) / 2
        
        logits = int_feats.matmul(W_out_prime)
        
        preds = logits.squeeze()
        
        return preds
    
    def make_symmetric(self, param):
        param_symm = param + param.T
        param_prime = (param_symm + torch.rot90(param, 2, (0, 1))) / 3
        return param_prime
        


#################
# Custom losses #
#################

class FocalLoss(nn.Module):
    
    def __init__(self, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, preds, target):
        
        target_colsums = torch.sum(target, dim=0)
        alpha = target_colsums / target.shape[0]
        
        pos_mask = target == 1
        neg_mask = target == 0
        alpha_mat = torch.zeros(*target.shape)
        alpha_mat += (pos_mask * alpha.repeat(target.shape[0], 1))
        alpha_mat += (neg_mask * (1 - alpha).repeat(target.shape[0], 1))
        
        BCE_loss = F.binary_cross_entropy_with_logits(preds, target,
                                                      reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = alpha_mat * (1-pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        return focal_loss
    