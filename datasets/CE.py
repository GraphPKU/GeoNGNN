from typing import List, Optional
from torch.utils.data import DataLoader
from typing import Optional, List, Callable
from torch_geometric.data import Data
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from datasets.transform_collate import collate_

from counterexamples.cube_reg_8 import generate_cube_plus_reg_8, sample_policys_cube_plus_reg_8
from counterexamples.reg_12 import generate_reg_12, sample_policys_12
from counterexamples.cube_cube import generate_cube_plus_cube, sample_policys_cube_plus_cube
from counterexamples.reg_20 import generate_reg_20, sample_policys_20
from counterexamples.augment2family import augment2family

generator_dict = {
    "cr8": generate_cube_plus_reg_8,
    "r12": generate_reg_12,
    "cc": generate_cube_plus_cube,
    "r20": generate_reg_20
}

policies_dict = {
    "cr8": sample_policys_cube_plus_reg_8,
    "r12": sample_policys_12,
    "cc": sample_policys_cube_plus_cube,
    "r20": sample_policys_20
}


class CE_dataset(InMemoryDataset):
    def __init__(
        self,
        name: str,
        transform: Optional[Callable] = None,
        draw: bool = False,
        save_dir: str = None,
        combine: bool = False,
    ):

        self.base_name, self.policy_order = name.split("-")
        self.policy_order = int(self.policy_order)
        self.transform = transform if transform is not None else lambda x: x
        self.draw = draw
        self.save_dir = save_dir
        self.combine = combine
        
        self.process()

    
    def process(self):
        if not self.combine:

            assert self.base_name in ["cr8", "r12", "cc", "r20"]
            
            generator = generator_dict[self.base_name]
            policies = policies_dict[self.base_name]
            assert self.policy_order < len(policies)
            policy_pair = policies[self.policy_order]
            left_policy, right_policy = policy_pair
            left_policy, right_policy = torch.tensor(left_policy, dtype=torch.bool), torch.tensor(right_policy, dtype=torch.bool) # (N, )
            rel_len = 0.5
            pos_list = generator(rel_len) # list
            pos = torch.tensor(pos_list, dtype=torch.float) # (N, 3)
            # graph1
            pos1 = pos[left_policy] # (n, 3)
            # graph2
            pos2 = pos[right_policy] # (n, 3)
            
            
        else:
            assert self.base_name in ["r12", "r20"]
            generator = generator_dict[self.base_name]
            policies = policies_dict[self.base_name]
            assert self.policy_order < len(policies)
            policy_pair = policies[self.policy_order]
            pos1, pos2 = augment2family(policy_pair, generator)
            pos1, pos2 = torch.tensor(pos1, dtype=torch.float), torch.tensor(pos2, dtype=torch.float) # (n, 3)

        pos1 = pos1 - pos1.mean(dim=0, keepdim=True)
        z1 = torch.ones(pos1.shape[0], dtype=torch.long) # (n, )
        y1 = torch.tensor([0], dtype=torch.float32) # (1, )
        
        pos2 = pos2 - pos2.mean(dim=0, keepdim=True)
        z2 = torch.ones(pos2.shape[0], dtype=torch.long) # (n, )
        y2 = torch.tensor([1], dtype=torch.float32) # (1, )
    
        assert pos1.shape[0] == pos2.shape[0]
        
        self.z = torch.cat([z1.unsqueeze(0), z2.unsqueeze(0)], dim=0) # (2, n)
        self.pos = torch.cat([pos1.unsqueeze(0), pos2.unsqueeze(0)], dim=0) # (2, n, 3)
        self.y = torch.cat([y1, y2], dim=0) # (2, )
        
        self._indices = torch.tensor([0, 1], dtype=torch.long)
        
        # get meta information
        dist_m1 = torch.cdist(pos1, pos1) # (n, n)        
        max1 = dist_m1.max()
        
        # rescale the graphs to make sure the max distance is 5.0
        coef = 5.0 / max1
        pos1, pos2 = pos1 * coef, pos2 * coef
        
        # reget meta information
        dist_m1 = torch.cdist(pos1, pos1) # (n, n)        
        max1, min1, mean1 = dist_m1.max(), dist_m1[dist_m1>0].min(), dist_m1.mean()
        print(f"max1: {max1}, min1: {min1}, mean1: {mean1}")

    def __getitem__(self, idx):
        data = Data(
            pos = self.pos[idx],
            z = self.z[idx],
            y = self.y[idx]
        )
        
        data = self.transform(data)
        return data
    
    
def CE_datawork(
    name: str,
    transform,
    max_neighbor,
    subgraph_cutoff: float = -1.0,
    cutoff: float = -1.0,
    extend_r=1.0,
    combine: bool = False,
    **kwargs,
    ):


    # get dataset and collate function.
    if transform is not None:
        transform_ = transform(
            dname=name,
            subgraph_cutoff=subgraph_cutoff,
            cutoff=cutoff,
            extend_r=extend_r,
            max_neighbor=max_neighbor
            )
    else:
        transform_ = None
    dataset = CE_dataset(name=name, transform=transform_, combine=combine)
        
    train_batch_size, val_batch_size, test_batch_size = 2, 2, 2
    
    
    
    
    # random_split dataset
    train_dataset, val_dataset, test_dataset = (
        dataset, 
        dataset,
        dataset
        )

    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = (
        DataLoader(train_dataset, num_workers=8, batch_size=train_batch_size, persistent_workers=True, shuffle=True, collate_fn=collate_),
        DataLoader(val_dataset, num_workers=4, batch_size=val_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_),
        DataLoader(test_dataset, num_workers=4, batch_size=test_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_),
    )

    return train_dataloader, val_dataloader, test_dataloader, None, None
