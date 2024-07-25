import torch
import torch.nn as nn
from layers.basic_layers import Residual, Dense
from layers.basis_layers import rbf_class_mapping
from torch_scatter import scatter
from torch.nn import Sequential
from layers.basic_layers import Envelope
EPS = 1e-10


class VD_Conv(nn.Module):
    def __init__(self, hidden_dim, activation_fn, ef_dim, agg):
        super().__init__()
        self.emb_mlp = nn.ModuleList(
                    [
                        Sequential(
                                Residual(
                                    mlp_num=2,
                                    hidden_dim=hidden_dim,
                                    activation_fn=activation_fn,
                                ), 
                            ) for _ in range(3)
                        ] 
                    )
        self.output_mlp = Sequential(
                Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,
                    ), 
                Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,
                    ), 
                )
            


        self.e_linear = Dense(
                    in_features=ef_dim,
                    out_features=hidden_dim,
                    bias=False,
                )
        self.conv_mlp = Sequential(
                Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,
                    ), 
                Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,
                    ), 
                )
        
        self.agg = agg

    def forward(self, scalar, ef, edge_index, C, conv_smooth):
            row, col = edge_index

            scalar_src, scalar_dst = [self.emb_mlp[j](scalar) for j in range(2)]
            conv_filter = self.e_linear(ef) * conv_smooth
            
            scalar_edge = scalar_dst[col] * conv_filter

            # Aggregate edge to scalar
            if self.agg == "mean":
                C = 1.
                
            conv = scatter(scalar_edge, row, dim=0, reduce=self.agg, dim_size=scalar_src.shape[0]) * C
            conv = self.conv_mlp(conv)
            
            scalar = scalar_src * conv # (N, hidden_dim)
            scalar = self.output_mlp(scalar) 
            
            return scalar
        
class Vanilla_DisGNN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 ef_dim,
                 is_inner,
                 activation_fn,
                 layer_num,
                 rbf,
                 cutoff,
                 extend_r=None,
                 agg="sum",
                 ):
        super().__init__()

        # Anchor distance encoding
        if is_inner:
            self.init_layer = Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn,
                        )
            self.DE_layer = nn.Sequential(
                    Dense(
                    in_features=ef_dim,
                    out_features=hidden_dim,
                    activation_fn=activation_fn
                ), 
                    Residual(
                        mlp_num=2,
                        hidden_dim=hidden_dim,
                        activation_fn=activation_fn,
                ), 
                )
            self.rbf_fn = rbf_class_mapping[rbf](
                num_rbf=ef_dim, 
                rbound_upper=extend_r,
                rbf_trainable=False,
            )
            
        self.VD_Convs = nn.ModuleList()

        for _ in range(layer_num):
            self.VD_Convs.append(
                VD_Conv(
                    hidden_dim=hidden_dim,
                    activation_fn=activation_fn,
                    ef_dim=ef_dim,
                    agg=agg,
                )
            )


        if not is_inner:
            self.pooling_MLP = nn.Sequential(
                    *nn.ModuleList([
                        Residual(
                            mlp_num=2,
                            hidden_dim=hidden_dim,
                            activation_fn=activation_fn,
                        )
                        for i in range(3)
                    ])
            )
            
        if is_inner:
            self.output_scalar = Residual(
                            mlp_num=2,
                            hidden_dim=hidden_dim,
                            activation_fn=activation_fn,
                        )
            
            self.pooling_MLP = nn.Sequential(
                    *nn.ModuleList([
                        Residual(
                            mlp_num=2,
                            hidden_dim=hidden_dim,
                            activation_fn=activation_fn,
                        )
                        for _ in range(3)
                    ])
            )
    
        self.p = 6
        self.envelope = Envelope(r_max=extend_r, p=self.p)
        self.layer_num = layer_num
        self.is_inner = is_inner
        self.extend_r = extend_r
        self.agg = agg
    def forward(
        self, scalar, ef,  edge_index, C, conv_smooth,
        dist=None, batch_index=None, subg_batch_index=None
    ):
        # Distance Incoding for inner GNN
        if self.is_inner:
            dist_rbf = self.rbf_fn(dist.unsqueeze(-1)) 
            scalar = self.DE_layer(dist_rbf) * self.init_layer(scalar)
        
        for i in range(self.layer_num):
            scalar = self.VD_Convs[i](
                scalar=scalar,
                ef=ef,
                edge_index=edge_index,
                C=C,
                conv_smooth=conv_smooth
            ) + scalar
        
        # Pooling
        if self.is_inner:
            scalar = self.output_scalar(scalar)

            subg_smooth = self.envelope(dist.unsqueeze(-1))
            scalar = scalar * subg_smooth
            
            subg_scalar = scatter(scalar, subg_batch_index, dim=0, reduce=self.agg)
            subg_scalar = self.pooling_MLP(subg_scalar)
    
            return scalar, subg_scalar
        else:
            graph_sum = scatter(scalar, batch_index, dim=0, reduce=self.agg)
            graph = self.pooling_MLP(graph_sum)
            
            return graph