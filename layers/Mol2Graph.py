import torch.nn as nn
from torch import Tensor
import torch
from layers.basis_layers import rbf_class_mapping
from layers.basic_layers import Envelope


class SparseMol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbound_upper: float,
                 max_z: int,
                 r_max: float = 10,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=False,
                    **kwargs
                )
        self.z_emb = nn.Embedding(max_z + 1, z_hidden_dim, padding_idx=0)
        self.p = 6
        self.envelope = Envelope(r_max=r_max, p=self.p)

    def forward(self, z: Tensor, pos: Tensor, edge_index: Tensor):
        '''
            z (N, ): atomic number
            pos (N, 3): atomic position
            edge_index (2, E): edge indices
        '''

        emb1 = self.z_emb(z)
        ev = pos[edge_index[0]] - pos[edge_index[1]] # (E, 3)
        el = torch.norm(ev, dim=-1, keepdim=True) # (E, 1)
        ef = self.rbf_fn(el) # (E, ef_dim)
        
        smooth_coef = self.envelope(el)
        
        return emb1, ef, smooth_coef




class Mol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbf_trainable: bool,
                 rbound_upper: float,
                 max_z: int,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=rbf_trainable,
                    **kwargs
                )
        self.z_emb = nn.Embedding(max_z + 1, z_hidden_dim, padding_idx=0)
        

    def forward(self, z: Tensor, pos: Tensor, **kwargs):
        '''
            z (B, N)
            pos (B, N, 3)
            emb1 (B, N, z_hidden_dim)
            ef (B, N, N, ef_dim)
            ev (B, N, N, 3)
        '''
        emb1 = self.z_emb(z)
        
        if kwargs.get("edge_index", None) is not None:
            edge_index = kwargs["edge_index"]
            ev = pos[edge_index[0]] - pos[edge_index[1]] # (E, 3)
            el = torch.norm(ev, dim=-1, keepdim=True) # (E, 1)
            ef = self.rbf_fn(el) # (E, ef_dim)
        else:
            B, N = z.shape[0], pos.shape[1]
            ev = pos.unsqueeze(2) - pos.unsqueeze(1)
            el = torch.norm(ev, dim=-1, keepdim=True)
            ef = self.rbf_fn(el.reshape(-1, 1)).reshape(B, N, N, -1)
        
        return emb1, ef

