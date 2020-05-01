import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .modules import AtomEmbed
from .modules import MultiHeadGraphAttn
from .modules import CondenseAttn
from .modules import FCContactPred
from .modules import OrderInvContPred
from .modules import FocalLoss


from .functions import pairwise_mat
from .functions import triu_condense
from .functions import triu_expand
from .functions import condense_res_tensors
from .functions import pairwise_3d

import matplotlib.pyplot as plt
import wandb


class GAT2D(pl.LightningModule):

    def __init__(self, embed_features, atom_out_features,
                 n_contact_channels, dropout, alpha, train_data, val_data,
                 test_data):
        super(GAT2D, self).__init__()
        
        # Properties
        self.embed_features = embed_features
        self.atom_out_features = atom_out_features
        self.n_contact_channels = n_contact_channels
        self.dropout = dropout
        self.alpha = alpha
        
        # Datasets
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # Model components
        self.embed = AtomEmbed(embed_features, scale_grad_by_freq=True)
    
        # Define number of graph conv layers and heads
        n_atom_layers = 5
        n_atom_heads = 3

        # Set up atom attention
        self.atom_attns = nn.ModuleList([])
        for i in range(n_atom_layers):
            if i == 0:
                attn_layer = MultiHeadGraphAttn(n_atom_heads, embed_features,
                                                atom_out_features,
                                                dropout, alpha)
            else:
                attn_layer = MultiHeadGraphAttn(n_atom_heads,
                                                n_atom_heads * atom_out_features,
                                                atom_out_features,
                                                dropout, alpha)
                
            self.atom_attns.append(attn_layer)
        
        self.conv1 = nn.Conv2d(n_atom_heads * atom_out_features * 2,
                               25, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(25,
                               25, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(25,
                               25, 3, stride=1, padding=1)
        
        self.conv_final = nn.Conv2d(25, n_contact_channels, 3, stride=1, padding=1)
        
        self.loss = FocalLoss(3)
        
        self.activations = {}

    def forward(self, x):
        atom_embed = self.embed(x['atom_nodes'].squeeze())
        
        atom_embed_update = atom_embed
        
        for layer in self.atom_attns:
            atom_embed_update = layer(atom_embed_update,
                                      x['atom_adj'].squeeze())
    
#         res_embed = x['mem_mat'].squeeze().matmul(atom_embed_update)
        res_embed = x['mem_mat'].squeeze().matmul(atom_embed_update)

        pairwise = pairwise_3d(res_embed).permute(2, 0, 1).unsqueeze(0)
        
        conv1_out = F.relu(self.conv1(pairwise))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        square_preds = self.conv_final(conv3_out)
        
        square_preds = square_preds + square_preds.permute(0, 1, 3, 2)
        
        preds = triu_condense(square_preds.squeeze().permute(1, 2, 0))

        atom_attns = [[head.f_attn for head in layer.attn_heads]
                      for layer in self.atom_attns]
        
        self.activations = {
            'atom_embed': atom_embed.detach().cpu().numpy(),
            'atom_embed_update': atom_embed_update.detach().cpu().numpy(),
            'atom_attn': atom_attns,
            'res_embed': res_embed.detach().cpu().numpy(),
            'combined': pairwise.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy()
        }
        
        return preds

    def training_step(self, batch, batch_nb):
        # REQUIRED
        y_hat = self.forward(batch)
        y = triu_condense(batch['res_contact'].squeeze())
        weights = triu_condense(batch['res_mask'].squeeze())
        
#         return {'loss': F.binary_cross_entropy_with_logits(y_hat[:, 1], y[:, 1])}

#         loss = F.binary_cross_entropy_with_logits(y_hat, y)
    
        loss = self.loss(y_hat[:, :3], y[:, :3])
    
        return {'loss': loss}
    
#         loss = F.binary_cross_entropy_with_logits(y_hat[1:3], y[1:3])
    
#         return {'loss': loss}

#     def validation_step(self, batch, batch_nb):
#         # OPTIONAL
# #         x, y = batch
# #         y_hat = self.forward(x)
# #         return {'val_loss': F.cross_entropy(y_hat, y)}
#         pass

#     def validation_end(self, outputs):
#         # OPTIONAL
# #         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
# #         return {'avg_val_loss': avg_loss}
#         pass

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        
        return torch.optim.SGD(parameters, lr=0.5, momentum=0.9)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_data, shuffle=True, num_workers=10, pin_memory=True)

#     @pl.data_loader
#     def val_dataloader(self):
#         # OPTIONAL
# #         return DataLoader(), batch_size=32)
#         pass

#     @pl.data_loader
#     def test_dataloader(self):
#         # OPTIONAL
# #         return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
#         pass

#     def on_batch_end(self):
#         wandb.log({key: plt.imshow(self.activations[key][:20]) for key in self.activations})
        

