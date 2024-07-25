import os
from typing import List, Optional
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from typing import Optional, List, Callable
from torch_geometric.data import Data
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
import torch
import numpy as np
from datasets.transform_collate import collate_



name_dict = {
    "Ac": "md22_Ac-Ala3-NHMe.npz",
    "DHA": "md22_DHA.npz",
    "AT": "md22_AT-AT.npz",
    "ATCG": "md22_AT-AT-CG-CG.npz",
    "Bu": "md22_buckyball-catcher.npz",
    "Do": "md22_double-walled_nanotube.npz",
    "St": "md22_stachyose.npz"
}

train_set_size = {
    "Ac": 6000,
    "DHA": 8000,
    "AT": 3000,
    "ATCG": 2000,
    "Bu": 600,
    "Do": 800,
    "St": 8000
}


class my_MD22(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.root = root
        path = os.path.join(root, name)
        self.path = path
        self.name = name
        super().__init__(path, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    @property
    def raw_paths(self) -> List[str]:
        return [os.path.join(self.path, "raw", name_dict[self.name])]
    
    def process(self):
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)

            z = torch.from_numpy(raw_data['z']).long()
            pos = torch.from_numpy(raw_data['R']).float()
            energy = torch.from_numpy(raw_data['E']).float()
            force = torch.from_numpy(raw_data['F']).float()

            data_list = []
            import tqdm
            for i in tqdm.tqdm(range(pos.size(0))):
                data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            torch.save(self.collate(data_list), processed_path)
            

    
def md22_datawork(
    root: str, 
    name: str,
    transform,
    batch_size: List[int],
    max_neighbor,
    subgraph_cutoff: float = -1.0,
    cutoff: float = -1.0,
    extend_r=1.0,
    ):

    
    train_set_num = train_set_size[name]
    data_point_num = [int(train_set_num * 0.95), train_set_num - int(train_set_num * 0.95)] 
    
    # get dataset and collate function.
    transform = transform(
        dname=name,
        subgraph_cutoff=subgraph_cutoff,
        cutoff=cutoff,
        extend_r=extend_r,
        max_neighbor=max_neighbor
        )
    dataset = my_MD22(root=root, name=name, transform=transform)
    
    
    # get meta data
    global_y_mean = dataset.data.energy.mean()
    global_y_std = dataset.data.energy.std()

    
    
    # get basic configs
    train_data_num, val_data_num = data_point_num
    test_data_num = len(dataset) - train_data_num - val_data_num
    train_batch_size, val_batch_size, test_batch_size = batch_size
    
    # random_split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_data_num, 
        val_data_num, 
        test_data_num]
        )
    
    

    
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = (
        DataLoader(train_dataset, num_workers=12, batch_size=train_batch_size, persistent_workers=True, shuffle=True, collate_fn=collate_),
        DataLoader(val_dataset, num_workers=12, batch_size=val_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_),
        DataLoader(test_dataset, num_workers=12, batch_size=test_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_),
    )
    
    return train_dataloader, val_dataloader, test_dataloader, global_y_mean, global_y_std

