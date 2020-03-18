import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .modules import AtomEmbed
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

import matplotlib.pyplot as plt
import wandb


class PGNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 anchorset_n, n_contact_channels, 
                 layer_num=2, train_data=None,
                 val_data=None, test_data=None):
        super(PGNN, self).__init__()
        
        # Parameters
        self.n_contact_channels = n_contact_channels
        self.layer_num = layer_num
        
        # Datasets
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # Model components
        self.embed = AtomEmbed(input_dim, scale_grad_by_freq=True)
        
        self.conv_atom_first = PGNN_layer(input_dim, hidden_dim)

        if layer_num>1:
            self.conv_atom_hidden = nn.ModuleList([PGNN_layer(hidden_dim,
                                                              hidden_dim)
                                                  for i in range(layer_num - 1)])
            
            self.conv_res_hidden = nn.ModuleList([PGNN_layer(hidden_dim,
                                                             hidden_dim)
                                                 for i in range(layer_num - 1)])
        
            
#         self.condense = CondenseAttn(hidden_dim,
#                                      hidden_dim,
#                                      0, 0.2)
            
        self.conv_out = PGNN_layer(hidden_dim, output_dim)
            
#         self.pred_contact = FCContactPred((anchorset_n + output_dim) * 2,
#                                           n_contact_channels,
#                                           layers=3)
        
        self.pred_contact = OrderInvContPred(anchorset_n,
                                             n_contact_channels, layers=5)
    
        init_bias = -torch.log(torch.FloatTensor([(1 - .001) / .001]))
        self.pred_contact_bias = nn.Parameter(init_bias, requires_grad=True)
        
        self.loss = FocalLoss(3)
        
        self.activations = {}


    def forward(self, data):
        
        data['atom_dist'] = generate_dists(data['atom_adj'])
        
        data['res_dist'] = generate_dists(data['res_adj'])

        data['atom_dist_max'], data['atom_dist_argmax'] = preselect_anchor(data['atom_adj'].squeeze().shape[0], data['atom_dist'])
        
        data['res_dist_max'], data['res_dist_argmax'] = preselect_anchor(data['res_adj'].squeeze().shape[0], data['res_dist'])
        
        
        #### Actual forward pass
        x = self.embed(data['atom_nodes'].squeeze())
        
        atom_embed = x.detach().cpu().numpy()
        
        x_position, x = self.conv_atom_first(x, data['atom_dist_max'], data['atom_dist_argmax'])
        x = F.relu(x) # Note: optional!
            
        for i in range(self.layer_num-1):
            _, x = self.conv_atom_hidden[i](x, data['atom_dist_max'], data['atom_dist_argmax'])
            x = F.relu(x) # Note: optional!
            
        atom_embed_update = x.detach().cpu().numpy()
            
        x = data['mem_mat'].squeeze().matmul(x) / data['mem_mat'].squeeze().mean(dim=1, keepdim=True)
        
        res_embed = x.detach().cpu().numpy()

        for i in range(self.layer_num-1):
            _, x = self.conv_res_hidden[i](x, data['res_dist_max'], data['res_dist_argmax'])
            x = F.relu(x) # Note: optional!
            
        res_embed_update = x.detach().cpu().numpy()
            
        x_position, x = self.conv_out(x, data['res_dist_max'], data['res_dist_argmax'])
        
        res_pos_embed = x_position.detach().cpu().numpy()
        
        x_position = F.normalize(x_position, p=2, dim=-1)
        
        res_embed_final = x_position
        
        combined = cat_pairwise(res_embed_final)
        
        preds = self.pred_contact(combined) + self.pred_contact_bias
        
        self.activations = {
            'atom_embed': atom_embed,
            'atom_embed_update': atom_embed_update,
            'res_embed': res_embed,
            'res_embed_update': res_embed_update,
            'res_pos_embed': res_pos_embed,
            'combined': combined.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy()
        }

        return preds

    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        y_hat = self.forward(batch)
        y = triu_condense(batch['res_contact'].squeeze())
        weights = triu_condense(batch['res_mask'].squeeze())
        loss = self.loss(y_hat[:, 1:3], y[:, 1:3])
    
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
        
        return torch.optim.SGD(parameters, lr=0.01, momentum=0.9)

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
