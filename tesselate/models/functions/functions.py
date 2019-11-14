import numpy as np
import torch


########################################
# Pairwise matrix generation functions #
########################################

def pairwise_mat(nodes, method='mean'):
    """
    Generate matrix for pairwise determination of interactions.
    
    Args:
    - nodes (torch.FloatTensor) - Tensor of node (n_nodes, n_features) features.
    - method (str) - One of 'sum' or 'mean' for combination startegy for
        pairwise combination matrix (default = 'mean').
        
    Returns:
    - torch.FloatTensor of shape (n_pairwise, n_nodes) than can be used used to
        combine feature vectors. Values are 1 if method == "sum" and 0.5 if
        method == "mean".
    """ 

    # Get the upper triangle indices
    triu = np.vstack(np.triu_indices(nodes.shape[0]))
    
    # Loop through all indices and add to list with 
    idxs = torch.from_numpy(triu).T
    
    # Convert to tensor
    combos = torch.zeros([idxs.shape[0], nodes.shape[0]]).scatter(1, idxs, 1)
    
    # Set to 0.5 if method is 'mean'
    if method == 'mean':
        combos *= 0.5
        
    return combos


def pairwise_3d(nodes):
    # Get the upper triangle indices
    repeated_nodes = nodes.unsqueeze(0).expand(nodes.shape[0], -1, -1)
    repeated_nodes2 = repeated_nodes.permute(1, 0, 2)
    
    return torch.cat((repeated_nodes, repeated_nodes2), dim=-1)


############################
# Upper triangle functions #
############################

def triu_condense(input_tensor):
    """
    Condense the upper triangle of a tensor into a 2d dense representation.
    
    Args:
    - input_tensor (torch.Tensor) - Tensor of shape (n, n, m).
    
    Returns:
    - Tensor of shape (n(n+1)/2, m) where elements along the third dimension in
        the original tensor are packed row-wise according to the upper
        triangular indices.
    """
    
    # Get upper triangle index info
    row_idx, col_idx = np.triu_indices(input_tensor.shape[0])
    row_idx = torch.LongTensor(row_idx)
    col_idx = torch.LongTensor(col_idx)
    
    # Return the packed matrix
    output = input_tensor[row_idx, col_idx, :]
    
    return output


def triu_expand(input_matrix):
    """
    Expand a dense representation of the upper triangle of a tensor into 
    a 3D squareform representation.
    
    Args:
    - input_matrix (torch.Tensor) - Tensor of shape (n(n+1)/2, m).
    
    Returns:
    - Tensor of shape (n, n, m) where elements along the third dimension in the
        original tensor are packed row-wise according to the upper triangular
        indices.
    """
    # Get the edge size n of the tensor
    n_elements = input_matrix.shape[0]
    n_chan = input_matrix.shape[1]
    n_res = int((-1 + np.sqrt(1 + 4 * 2 * (n_elements))) / 2)
    
    # Get upper triangle index info
    row_idx, col_idx = np.triu_indices(n_res)
    row_idx = torch.LongTensor(row_idx)
    col_idx = torch.LongTensor(col_idx)
    
    # Generate the output tensor
    output = torch.zeros((n_res, n_res, n_chan))
    
    # Input the triu values
    for i in range(n_chan):
        i_tens = torch.full((len(row_idx),), i, dtype=torch.long)
        output.index_put_((row_idx, col_idx, i_tens), input_matrix[:, i])
    
    # Input the tril values
    for i in range(n_chan):
        i_tens = torch.full((len(row_idx),), i, dtype=torch.long)
        output.index_put_((col_idx, row_idx, i_tens), input_matrix[:, i])
    
    return output


########################################
# Complex node summarization functions #
########################################

def summarize_res_tensor(tensor, _count=True, _mean=True, _sum=True,
                         _max=True, _min=True, _std=True):
    """
    Calculate a dense summarization of node features on condensation.
    
    Args:
    - tensor (torch.FloatTensor) - Tensor (n_nodes, n_features) to summarize. 
    - _count (bool) - Calculate the count of nodes in the metanode 
        (default = True)
    - _mean (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    - _sum (bool) - Calculate the  of the nodes in the metanode
        (default = True).   
    - _max (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    - _min (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    - _std (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    
    Returns:
    - torch.FloatTensor (1, n_summary_features) of the summary of the
        distribution of nodes.
    """
    
    # List to collect tensors of distributional statistics
    stats = []
    
    # Get the appropriate summary stat if True
    if _count:
        stats.append(torch.tensor([[len(tensor)]], dtype=torch.float))
        
    if _mean:
        stats.append(tensor.mean(dim=0, keepdim=True))
        
    if _sum:
        stats.append(tensor.sum(dim=0, keepdim=True))
        
    if _max:
        stats.append(tensor.max(dim=0, keepdim=True)[0])
        
    if _min:
        stats.append(tensor.min(dim=0, keepdim=True)[0])
        
    if _std:
        stats.append(tensor.std(dim=0, keepdim=True))
        
    return torch.cat(stats, dim=1)


def condense_res_tensors(embeddings, mem_mat, _count=True, _mean=True,
                         _sum=True, _max=True, _min=True, _std=True):
    """
    Calculate a dense summarization of all metanodes according to a membership
    matrix.
    
    Args:
    - embeddings (torch.FloatTensor) - Tensor (n_nodes, n_features) of node
        embeddings to summarize.
    - mem_mat (torch.FloatTensor) - Tensor (n_metanodes, n_nodes) of boolean
        membership for each node to each metanode (1 if member, 0 otherwise).
    - _count (bool) - Calculate the count of nodes in the metanode 
        (default = True)
    - _mean (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    - _sum (bool) - Calculate the  of the nodes in the metanode
        (default = True).   
    - _max (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    - _min (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    - _std (bool) - Calculate the  of the nodes in the metanode
        (default = True).
    
    Returns:
    - torch.FloatTensor (n_metanodes, n_summary_features) of the summary of the
        distribution of nodes.
    """

    # List to hold summary rows
    res_tensors = []
    
    # Get summary for each metanode
    for i in mem_mat.bool():
        res_tensor = summarize_res_tensor(embeddings[i], _count,
                                          _mean, _sum, _max, _min, _std)
        
        res_tensors.append(res_tensor)
    
    return torch.cat(res_tensors, dim=0)


####################################
# Pairwise concatenation functions #
####################################

def cat_pairwise(embeddings):
    
    triu = np.vstack(np.triu_indices(embeddings.shape[0])).T
    
    node1 = []
    node2 = []

    for i, j in triu:
        node1.append(embeddings[i])
        node2.append(embeddings[j])

    node1 = torch.stack(node1, dim=0)
    node2 = torch.flip(torch.stack(node2, dim=0), dims=(1,))

    return torch.cat((node1, node2), dim=1)


###################
# P-GNN functions #
###################

def generate_dists(adj_mat):
    adj_mask = adj_mat == 0
    
    dist = adj_mat - torch.eye(adj_mat.shape[0])
    dist = 1 / (dist + 1)
    dist[adj_mask] = 0
    
    return dist.squeeze()


def get_dist_max(anchorset_id, dist):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id)))
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long()
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = dist_argmax_temp
    return dist_max, dist_argmax


def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id


def preselect_anchor(n_nodes, dists):
    anchorset_id = get_random_anchorset(n_nodes, c=1)
    return get_dist_max(anchorset_id, dists)


    