import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .modules import AtomEmbed, AtomOneHotEmbed
from .modules import PGNN_layer
from .modules import CondenseAttn
from .modules import FCContactPred
from .modules import OrderInvContPred
from .modules import FocalLoss

from .functions import pairwise_mat
from .functions import triu_condense
from .functions import condense_res_tensors
from .functions import cat_pairwise
from .functions import generate_dists
from .functions import preselect_anchor
from .functions import pairwise_3d

import matplotlib.pyplot as plt
import wandb


class PGNN2D(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 anchorset_n, n_contact_channels, 
                 layer_num=2, train_data=None,
                 val_data=None, test_data=None):
        super(PGNN2D, self).__init__()
        
        # Parameters
        self.n_contact_channels = n_contact_channels
        self.layer_num = layer_num
        
        # Datasets
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # Model components
#         self.embed = AtomEmbed(input_dim, scale_grad_by_freq=True)
        self.embed = AtomOneHotEmbed()
        
        self.conv_atom_first = PGNN_layer(input_dim, hidden_dim)

        if layer_num>1:
            self.conv_atom_hidden = nn.ModuleList([PGNN_layer(hidden_dim,
                                                              hidden_dim)
                                                  for i in range(layer_num - 1)])
        
        self.conv1 = nn.Conv2d(anchorset_n * 2,
                               25, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(25,
                               25, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(25,
                               25, 3, stride=1, padding=1)
        
        self.conv_final = nn.Conv2d(25, n_contact_channels, 3, stride=1, padding=1)
        
        self.loss = FocalLoss(3, reduction='sum')
        
        self.activations = {}


    def forward(self, data):
        
        data['atom_dist'] = generate_dists(data['atom_adj'])

        data['atom_dist_max'], data['atom_dist_argmax'] = preselect_anchor(data['atom_adj'].squeeze().shape[0], data['atom_dist'])
        
        #### Actual forward pass
        x = self.embed(data['atom_nodes'].squeeze())
        
        atom_embed = x.detach().cpu().numpy()
        
        x_position, x = self.conv_atom_first(x, data['atom_dist_max'], data['atom_dist_argmax'])
        x = F.relu(x) # Note: optional!
            
        for i in range(self.layer_num-1):
            
            data['atom_dist'] = generate_dists(data['atom_adj'])

            data['atom_dist_max'], data['atom_dist_argmax'] = preselect_anchor(data['atom_adj'].squeeze().shape[0], data['atom_dist'])
            
            x_position, x = self.conv_atom_hidden[i](x, data['atom_dist_max'], data['atom_dist_argmax'])
            x = F.relu(x) # Note: optional!
            
        atom_embed_update = x_position.detach().cpu().numpy()
            
        x_position = data['mem_mat'].squeeze().matmul(x_position)
        
        res_embed = x_position.detach().cpu().numpy()
        res_embed.shape
        
        pairwise = pairwise_3d(x_position).permute(2, 0, 1).unsqueeze(0)
        
        conv1_out = F.relu(self.conv1(pairwise))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        square_preds = self.conv_final(conv3_out)
        
        square_preds = square_preds + square_preds.permute(0, 1, 3, 2)
        
        preds = triu_condense(square_preds.squeeze().permute(1, 2, 0))
        
        self.activations = {
            'atom_embed': atom_embed,
            'atom_embed_update': atom_embed_update,
            'res_embed': res_embed,
            'combined': pairwise.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy()
        }

        return preds

    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        y_hat = self.forward(batch)
        y = triu_condense(batch['res_contact'].squeeze())
#         weights = triu_condense(batch['res_mask'].squeeze())

        loss = self.loss(y_hat, y)
    
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
        
        return torch.optim.SGD(parameters, lr=1e-6, momentum=0.9)

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

