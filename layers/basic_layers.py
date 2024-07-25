import torch.nn as nn
import torch
from utils.initializers import he_orthogonal_init

class Dense(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        activation_fn: torch.nn.Module = None,
        activation_first: bool = False,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        
        self.reset_parameters()
        
        self.activation_first = activation_first
        self._activation = torch.nn.Identity() if activation_fn is None else activation_fn
        
    def reset_parameters(self):
        if not self.in_features == 1:
            he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, x_type: torch.Tensor = None):
        if self.activation_first:
            x = self._activation(x)
            x = self.linear(x)
        else:
            x = self.linear(x)
            x = self._activation(x)
        return x


class Residual(nn.Module):
    def __init__(
        self,
        mlp_num: int,
        hidden_dim: int,
        activation_fn: torch.nn.Module = None,
        bias: bool = True,
        activation_first: bool = True,
    ):
        super().__init__()
        assert mlp_num == 2
        
        self.mlp1 = Dense(hidden_dim, hidden_dim, bias=bias, activation_fn=activation_fn, activation_first=activation_first)
        self.mlp2 = Dense(hidden_dim, hidden_dim, bias=bias, activation_fn=activation_fn, activation_first=activation_first)
            
    def forward(self, x: torch.Tensor):
        y = self.mlp1(x)
        y = self.mlp2(y)
        
        return x + y
    
    



class Envelope(nn.Module):
    '''
    Gasteiger, Johannes, Janek Groß, and Stephan Günnemann. "Directional message passing for molecular graphs." arXiv preprint arXiv:2003.03123 (2020).
    '''
    def __init__(self, r_max, p):
        super().__init__()
        self.r_max = r_max
        self.p = p
    def forward(self, x):
        smooth_coef = (
                    1.0
                    - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                    + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                    - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
            )
        smooth_coef[x >= self.r_max] = 0.0
        return smooth_coef
