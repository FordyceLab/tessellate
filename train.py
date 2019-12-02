from pytorch_lightning import Trainer
from tesselate.models.vanilla_gat import VanillaGAT
from tesselate.datasets import TesselateDataset

if __name__ == '__main__':
    
    train_data = TesselateDataset('test2.txt', graph_dir='data/graphs',
                     contacts_dir='data/contacts', return_data='all')
    
    model = VanillaGAT(embed_features=10, atom_out_features=10, res_out_features=10,
                       n_contact_channels=8, dropout=0, alpha=0.2, train_data=train_data,
                       val_data=None, test_data=None)
    
    trainer = Trainer()
    trainer.fit(model)