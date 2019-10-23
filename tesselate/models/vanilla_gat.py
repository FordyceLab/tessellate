import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .modules.modules import AtomEmbed
from .modules.modules import GraphAttn
from .modules.modules import FCContactPred

from .functions.functions import pairwise_mat
from .functions.functions import triu_condense
from .functions.functions import condense_res_tensors


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
        self.atom_attn = GraphAttn(embed_features, atom_out_features, dropout, alpha)
        self.res_attn = GraphAttn(atom_out_features, res_out_features, dropout, alpha)
        self.pred_contact = FCContactPred(res_out_features, n_contact_channels)

    def forward(self, x):
        atom_embed = self.embed(x['atom_nodes'].squeeze())
        atom_attn = self.atom_attn(atom_embed, x['atom_adj'].squeeze())
        res_embed = x['mem_mat'].squeeze().matmul(atom_attn)
        res_attn = self.res_attn(res_embed, x['res_adj'].squeeze())
        pairwise = pairwise_mat(res_attn, method='sum')
        
#         combined = condense_res_tensors(res_attn, pairwise, _count=False, _mean=True,
#                                         _sum=True, _max=True, _min=True, _std=False)
        
        combined = pairwise.matmul(res_attn)
#         print(torch.round(combined[15:20] * 10**2) / (10**2))
        

#         print(combined.shape)
        

        preds = self.pred_contact(combined)
        
        return preds

    def training_step(self, batch, batch_nb):
        # REQUIRED
        y_hat = self.forward(batch)
        y = triu_condense(batch['res_contact'].squeeze())
#         weights = triu_condense(batch['res_mask'].squeeze())
        
        n = 10
    
#         print(torch.min(y_hat), torch.max(y_hat))
        print(torch.round(y_hat[:n] * 10**2) / (10**2))
        print(y[:n])
        
#         return {'loss': F.binary_cross_entropy(y_hat, y, weight=weights)}
    
        loss = F.binary_cross_entropy(y_hat[:n], y[:n])
        loss = .01 * (1 / torch.sum(torch.std(y_hat[:n], dim=0)))
    
        return {'loss': loss}

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
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

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