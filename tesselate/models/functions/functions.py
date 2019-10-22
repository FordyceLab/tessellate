import numpy as np
import torch


def pairwise_mat(nodes, method='mean'):
    """
    Generate matrix for pairwise determination of interactions.
    
    Args:
    - nodes (torch.FloatTensor) - Tensor of node features
    - method (str) - One of "sum" or "mean" for combination
        startegy for pairwise combination matrix
        
    Returns:
    - torch.FloatTensor of shape (n_pairwise_combos, n_nodes) than
        can be used used to combine feature vectors. Values are
        1 if method == "sum" and 0.5 if method == "mean".
    """ 

    # Get the upper triangle indices
    triu = np.vstack(np.triu_indices(nodes.shape[0]))
    
    # Loop through all indices and add to list with 
    idxs = torch.from_numpy(triu).T
    
    # Convert to tensor
    combos = torch.zeros([idxs.shape[0], nodes.shape[0]]).scatter(1, idxs, 1)
    
    if method == 'mean':
        combos *= 0.5
        
    return combos


def triu_condense(input_tensor):
    """
    Condense the upper triangle of a tensor into a 2d
    dense representation.
    
    Args:
    - input_tensor (torch.Tensor) - Tensor of shape (n, n, m)
    
    Returns:
    - Tensor of shape (n(n+1)/2, m) where elements along the
        third dimension in the original tensor are packed
        row-wise according to the upper triangular indices.
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
    Condense the upper triangle of a tensor into a 2d
    dense representation.
    
    Args:
    - input_matrix (torch.Tensor) - Tensor of shape (n(n+1)/2, m)
    
    Returns:
    - Tensor of shape (n, n, m) where elements along the
        third dimension in the original tensor are packed
        row-wise according to the upper triangular indices.
    """
    # Get the edge size n of the tensor
    num_elements = input_matrix.shape[0]
    n = (-1 + np.sqrt(1 + 4(num_elements))) / 2
    
    # Get upper triangle index info
    row_idx, col_idx = np.triu_indices(input_tensor.shape[0])
    row_idx = torch.LongTensor(row_idx)
    col_idx = torch.LongTensor(col_idx)
    
    # Input the triu values
    output = torch.zeros().index_put_((row_idx, col_idx), input_matrix)
    
    # Get lower triangle index info
    row_idx, col_idx = np.tril_indices(n)
    row_idx = torch.LongTensor(row_idx)
    col_idx = torch.LongTensor(col_idx)
    
    # Input the tril values
    output = output.index_put_((row_idx, col_idx), input_matrix)
    
    return output
