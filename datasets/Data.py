from torch_geometric.data import Data
from typing import Any

class NestedData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ['subg_edge_index']:
            return self.subg_size.sum()
        if key in ['batch_index']:
            return 1
        if key in ["subg_batch_index"]:
            return self.K
        if key in ['subg_node_index', 'subg_node_center_index', 'edge_index']:
            return self.z.shape[0]
        if key in ["subg_node_label"]:
            return 0
    
        infer = super().__inc__(key, value, *args, **kwargs)
        return infer
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ['subg_edge_index', 'edge_index']:
            return -1
        if key in ['batch_index', 'subg_node_label', 'subg_node_index', 'subg_node_center_index', "subg_batch_index"]:
            return 0
        
        infer = super().__cat_dim__(key, value, *args, **kwargs)
        return infer
    
    
    
class SubgData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ['subg_edge_index']:
            return self.subg_size
        if key in ['subg_batch_index']:
            return 1
        if key in ['subg_node_label', 'subg_node_index', 'subg_node_center_index']:
            return 0
        
        infer = super().__inc__(key, value, *args, **kwargs)
        return infer
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ['subg_edge_index']:
            return -1
        if key in ['subg_node_label', 'subg_node_index', 'subg_node_center_index', "subg_batch_index"]:
            return 0
        
        infer = super().__cat_dim__(key, value, *args, **kwargs)
        return infer

