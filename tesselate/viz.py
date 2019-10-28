import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

######################
# ROC and PRC curves #
######################

def calc_metric_curve(preds, target, curve_type, squareform=False):
    """
    Calculate ROC or PRC curves and area for the predicted contact channels.
    
    Args:
    - preds (np.ndarray) - Numpy array of model predictions either of form
        (n_res, n_res, n_chan) or (n_res * [n_res - 1] / 2, n_chan).
    - target (np.ndarray) - Numpy array of target values either of form
        (n_res, n_res, n_chan) or (n_res * [n_res - 1] / 2, n_chan),
        must match form of preds.
    - curve_type (str) - One of 'ROC' or 'PRC' to denote type of curve.
    - squareform (bool) - True if tensors are of shape (n_res, n_res, n_chan),
        False if they are of shape (n_res * [n_res - 1] / 2, n_chan)
        (default = True).
    
    Returns:
    - Tuple of x, y, and AUC values to be used for plotting the curves
        using plot_curve metric.
    """
    
    # Get correct curve function
    if curve_type.upper() == 'ROC':
        curve_func = roc_curve
    elif curve_type.upper() == 'PRC':
        curve_func = precision_recall_curve
        
    # Generate dicts to hold outputs from curve generation functions
    x = dict()
    y = dict()
    auc_ = dict()
    
    # Handle case of squareform matrix (only get non-redundant triu indices)
    if squareform:
        indices = np.triu_indices(target.shape[0])

    # For each channel
    for i in range(target.shape[-1]):
        
        # Handle case of squareform
        if squareform:
            var1, var2, _ = curve_func(target[:, :, i][indices], preds[:, :, i][indices])
            
        # Handle case of pairwise
        else:
            var1, var2, _ = curve_func(target[:, i], preds[:, i])
        
        # Assign outputs to correct dict for plotting
        if curve_type.upper() == 'ROC':
            x[i] = var1
            y[i] = var2
        elif curve_type.upper() == 'PRC':
            x[i] = var2
            y[i] = var1
        
        # Calc AUC
        auc_[i] = auc(x[i], y[i])
        
    return (x, y, auc_)


def plot_curve_metric(x, y, auc, curve_type, title=None, labels=None):
    """
    Plot ROC or PRC curves per output channel.
    
    Args:
    - x (dict) - Dict of numpy arrays for values to plot on x axis.
    - y (dict) - Dict of numpy arrays for values to plot on x axis.
    - auc (dict) - Dict of numpy arrays for areas under each curve.
    - curve_type (str) - One of 'ROC' or 'PRC' to denote type of curve.
    - title
    - labels
    
    Returns:
    - pyplot object of curves. 
    """
    
    # Generate figure
    plt.figure()
    
    # Linetype spec
    lw = 2
    curve_type = curve_type.upper()
    
    # Get the number of channels being plotted
    n_chan = len(x)
    
    # Check to make sure the labels are the right length
    if len(labels) != n_chan:
        raise ValueError('Number of labels ({}) does not match number of prediction channels ({}).'.format(len(labels), n_chan))
    
    # Make labels numeric if not provided
    if labels is None:
        labels = list(range(n_chan))
    
    # Get a lit of colors for all the channels
    color_list = plt.cm.Set1(np.linspace(0, 1, n_chan))
    
    # Plot each line
    for i, color in enumerate(color_list):
        plt.plot(x[i], y[i], color=color,
                 lw=lw, label='{} (area = {:0.2f})'.format(labels[i], auc[i]))
        
    # Add labels and diagonal line for ROC
    if curve_type == 'ROC':
        xlab = 'FPR'
        ylab = 'TPR'
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.legend(loc="lower right")

    # Add labels for PRC 
    elif curve_type == 'PRC':
        xlab = 'Recall'
        ylab = 'Precision'
        plt.legend(loc="lower left")
    
    # Extend limits, add labels and title
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    if title is not None:
        plt.title('{} for {}'.format(curve_type, title))
    else:
        plt.title('{}'.format(curve_type))
    
    return plt

def plot_curve(preds, target, curve_type, title=None, labels=None,
               squareform=False):
    
    calc_metric_curve(preds, target, curve_type, squareform=False)

##################################
# Intermediate outputs/gradients #
##################################
