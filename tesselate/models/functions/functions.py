import numpy as np
import torch

def pairwise_mat(nodes, method='mean'):
    """
    Generate matrix for pairwise determination of interactions.
    
    Args:
    - 
    - method 
    
    """ 
    indices = torch.from_numpy(np.vstack(np.triu_indices(nodes.shape).transpose()))
    combos = torch.sparse.FloatTensor(torch.ones(indices.shape[1]),
                                      torch.Size(indices.shape[1], indices.shape[1])).to_dense()
    
    if method == 'mean':
        combos *= 0.5
        
    return combos