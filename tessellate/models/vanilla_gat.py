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
from .functions import cat_pairwise

import matplotlib.pyplot as plt
import wandb


class VanillaGAT(pl.LightningModule):

    def __init__(self, embed_features, atom_out_features, res_out_features,
                 n_contact_channels, dropout, alpha, train_data, val_data,
                 test_data):
        super(VanillaGAT, self).__init__()
        
        # Properties
        self.embed_features = embed_features
        self.atom_out_features = atom_out_features
        self.res_out_features = res_out_features
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
        
        n_res_layers = 3
        n_res_heads = 3

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
            
            
        # Set up condensation
        self.condense = CondenseAttn(n_atom_heads * atom_out_features,
                                     n_atom_heads * atom_out_features,
                                     dropout, alpha)
        
        # Set up res attention
        self.res_attns = nn.ModuleList([])
        for i in range(n_res_layers):
            if i == 0:
                attn_layer = MultiHeadGraphAttn(n_res_heads,
                                                n_atom_heads * atom_out_features,
                                                res_out_features, dropout,
                                                alpha)
            else:
                attn_layer = MultiHeadGraphAttn(n_res_heads,
                                                n_res_heads * res_out_features,
                                                res_out_features, dropout,
                                                alpha)
                
            self.res_attns.append(attn_layer)
        
        self.activation = nn.LeakyReLU(self.alpha)
        
#         self.pred_contact = OrderInvContPred(n_res_heads * res_out_features,
#                                              n_contact_channels)
        
        self.pred_contact = FCContactPred(n_res_heads * res_out_features * 2,
                                          n_contact_channels,
                                          layers=5)
    
        init_bias = -torch.log(torch.FloatTensor([(1 - .001) / .001]))
        self.pred_contact_bias = nn.Parameter(init_bias, requires_grad=True)
        
        self.loss = FocalLoss(3)
        
        self.activations = {}

    def forward(self, x):
        atom_embed = self.embed(x['atom_nodes'].squeeze())
        
        atom_embed_update = atom_embed
        
        for layer in self.atom_attns:
            atom_embed_update = layer(atom_embed_update,
                                      x['atom_adj'].squeeze())
    
#         res_embed = x['mem_mat'].squeeze().matmul(atom_embed_update)
        res_embed = self.condense(atom_embed_update, x['mem_mat'].squeeze())

        #res_embed = x['res_contact'].squeeze()[:, :, 1]
        res_embed_update = res_embed
        
        
        # Do final layer with larger receptive field
        
        adj = x['res_adj'].squeeze()
        
        for layer in self.res_attns[:-1]:
#             print(adj)
            
            res_embed_update = layer(res_embed_update, adj)
            combined = cat_pairwise(res_embed_update)
            preds = self.pred_contact(combined)
            adj = torch.sigmoid(triu_expand(preds)[:, :, 1] + x['res_adj'].squeeze())
            
        final_layer = self.res_attns[-1]
        
#         res_adj = x['res_adj'].squeeze()
#         wide_field = res_adj
        
#         for i in range(0):
#             wide_field = wide_field.matmul(res_adj)
            
#         wide_field = (wide_field > 0).float()

        wide_field = adj
    
#         print(adj)
            
        res_embed_update = final_layer(res_embed_update, wide_field)
        
#         print(res_embed_update)
        
#         res_attn = self.res_attn(res_embed, x['res_adj'].squeeze())
        
        # Get the output
        # (n_nodes, out_features)
#         res_embed_update = self.activation(torch.matmul(res_attn, res_embed)).squeeze()
        
#         for i in range(4):
#             res_attn = self.res_attn(res_embed_update, x['res_adj'].squeeze())

#             # Get the output
#             # (n_nodes, out_features)
#             res_embed_update = self.activation(torch.matmul(res_attn, res_embed)).squeeze()
        
        
#         pairwise = pairwise_mat(res_embed_update, method='sum')
#         combined = condense_res_tensors(res_attn, pairwise, _count=False, _mean=True,
#                                         _sum=True, _max=True, _min=True, _std=False)
        
#         combined = pairwise.matmul(res_embed_update)
#         print(torch.round(combined[15:20] * 10**2) / (10**2))
        
    
        combined = cat_pairwise(res_embed_update)
        
#         print(combined)
        

#         print(combined.shape)
        
        preds = self.pred_contact(combined) + self.pred_contact_bias
#         preds = self.pred_contact(combined)

#         print(len(self.res_attns))

#         for i, layer in enumerate(self.res_attns):
#             for j, head in enumerate(layer.attn_heads):
#                 print(i, j)
#                 print(hasattr(head, 'f_attn'))

        atom_attns = [[head.f_attn for head in layer.attn_heads]
                      for layer in self.atom_attns]
        res_attns = [[head.f_attn for head in layer.attn_heads]
                      for layer in self.res_attns]
        
        
        
        self.activations = {
            'atom_embed': atom_embed.detach().cpu().numpy(),
            'atom_embed_update': atom_embed_update.detach().cpu().numpy(),
            'atom_attn': atom_attns,
            'res_embed': res_embed.detach().cpu().numpy(),
            'res_embed_update': res_embed_update.detach().cpu().numpy(),
            'res_attn': res_attns,
#             'pairwise': pairwise.detach().cpu().numpy(),
            'combined': combined.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy()
        }
#         for key in self.activations:
#             print(self.activations[key].shape)
        
        return preds

    def training_step(self, batch, batch_nb):
        # REQUIRED
        y_hat = self.forward(batch)
        
#         vx = embeds[:, 0] - torch.mean(embeds[:, 0])
#         vy = embeds[:, 1] - torch.mean(embeds[:, 1])
#         cost = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))**2
        
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
        
