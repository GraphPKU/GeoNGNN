from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Batch
from datasets.Data import NestedData, SubgData
from utils.atom_mass import atom_mass_dict

class subg_transform(BaseTransform):
    def __init__(self, dname, subgraph_cutoff, cutoff, extend_r, max_neighbor):
        self.dname = dname
        self.subgraph_cutoff = subgraph_cutoff
        self.cutoff = cutoff
        self.extend_r = extend_r
        self.max_neighbor = max_neighbor
        
        
    def __call__(self, data):
        '''
        To add origin2part and K number to data
        ''' 
        
        N = data.pos.shape[0]
        data.K = N

        '''
        To calculate subgraph related info and wrap the data as NestedData
        '''
        
        subg_datas = []
        for i in range(data.K):
            subg_data = self.subg_cal(
                data=data,
                subgraph_radius=self.extend_r,
                center_idx=i,
            )
            subg_datas.append(subg_data)
        loader = DataLoader(subg_datas, batch_size=len(subg_datas), shuffle=False)
        subg_datas_batched = next(iter(loader))



        nested_data = NestedData()
        for var in data.keys:
            nested_data[var] = data[var]
        for var in subg_datas_batched.keys:
            nested_data[var] = subg_datas_batched[var]
        nested_data["mass"] = torch.tensor([atom_mass_dict[zi.item()] for zi in data.z], dtype=torch.float32)
        '''
        To add global edge_index to data
        '''

        if self.cutoff is None:
            self.cutoff = torch.inf
        dist_matrix = (data.pos.unsqueeze(0) - data.pos.unsqueeze(1)).norm(dim=-1) # (N, N)
        global_mask = (dist_matrix <= self.cutoff) * (dist_matrix > 0.)
        edge_index = global_mask.nonzero(as_tuple=False).t() # (2, E)
        
        nested_data["edge_index"] = edge_index
        

        '''
        To add batch_index to data
        '''
        nested_data.batch_index = torch.zeros(data.pos.shape[0], dtype=torch.long)
        
        return nested_data
    
        
    def subg_cal(self, data, subgraph_radius, center_idx):
        node_num = data.pos.shape[0]

        dist = (data.pos - data.pos[center_idx].view(1, -1)).norm(dim=1) # (node_num)
        dist_rank = torch.argsort(torch.argsort(dist))
        candidate_indices = dist_rank < self.max_neighbor
        ori_mask = dist <= subgraph_radius
        mask = ori_mask & candidate_indices

        subg_node_index = torch.arange(node_num, dtype=torch.long)[mask]
        subg_size_origin = ori_mask.sum()
        subg_size = mask.sum()
        
        subg_z = data.z[mask] # (subg_node_num)
        
        self_index = mask[:center_idx].sum()
        
        subg_node_label = torch.ones_like(subg_z)
        subg_node_label[self_index] = 2
        
        
        subg_node_index = torch.arange(node_num, dtype=torch.long)[mask] # To get pos in real time, for maintaining the grad.
        subg_node_center_index = torch.ones_like(subg_node_index) * center_idx
        
        poss = data.pos[subg_node_index] # (N, 3)
        distance_matrix = (poss.unsqueeze(0) - poss.unsqueeze(1)).norm(dim=-1) # (N, N)
        edge_candidates = (distance_matrix <= self.subgraph_cutoff) * (distance_matrix > 0.)
        subg_edge_index = edge_candidates.nonzero(as_tuple=False).t() # (2, M)
        
        subg_batch_index = torch.zeros(subg_size, dtype=torch.long)


        subg_data = SubgData()
        for var in ['subg_z', 'subg_node_label', 'subg_edge_index', 
                    'subg_node_index', 'subg_node_center_index', "subg_batch_index", "subg_size"]:
            subg_data[var] = locals()[var]
        
        
        return subg_data
    


    
            

    
def collate_(data_list, name=None):
    if name is not None:
        for i in range(len(data_list)):
            data_list[i].y = data_list[i].y[0, int(name)]
    
    data = Batch.from_data_list(data_list)
    
    return data

def transform_collate_(data_list, transform):
    new_data_list = []
    for data in data_list:
        new_data = transform(data)
        new_data_list.append(new_data)
    data = Batch.from_data_list(new_data_list)
    
    return data
