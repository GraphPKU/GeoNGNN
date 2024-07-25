from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
from typing import List, Optional
import os
from typing import Optional, List, Callable
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from datasets.transform_collate import collate_

atom_type2z = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
}



class triBPA_dataset(InMemoryDataset):
    def __init__(        
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
        ):
        self.root = root
        self.name = name
        self.transform = transform
        self.load_data()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        return [self.name + '_data.pt']
    def raw_file_names(self) -> List[str]:
        return [self.name + ".xyz"]
    
    
    def load_data(self):
        
        file_name = self.name + ".xyz"
        
        # read xyz file
        def read_xyz(file_path):    
            with open(file_path, 'r') as f:
                lines = f.readlines()
            molecules = []
            atom_num = int(lines[0].strip())

            assert len(lines) % (atom_num + 2) == 0
            molecule_num = len(lines) // (atom_num + 2)
            
            for molecule_idx in range(molecule_num):
                atoms = []
                for atom_idx in range(atom_num + 2):
                    line = lines[molecule_idx * (atom_num + 2) + atom_idx]
                    if atom_idx == 0:
                        assert int(line.strip()) == atom_num
                    elif atom_idx == 1:
                        comment = line.strip()
                    else:
                        line_list = line.strip().split()
                        assert len(line_list) == 7
                        atom_type = line_list[0]
                        x, y, z = float(line_list[1]), float(line_list[2]), float(line_list[3])
                        fx, fy, fz = float(line_list[4]), float(line_list[5]), float(line_list[6])
                        atoms.append([atom_type, x, y, z, fx, fy, fz])
                assert comment is not None
                molecules.append((comment, atoms))
                comment = None
                
            return molecules
        
        molecules = read_xyz(self.raw_paths[0])
        
        # process molecules
        molecule_num = len(molecules)
        atom_num = len(molecules[0][1])
        
        z = torch.empty((molecule_num, atom_num), dtype=torch.long)
        pos = torch.empty((molecule_num, atom_num, 3), dtype=torch.float)
        force = torch.empty((molecule_num, atom_num, 3), dtype=torch.float)
        energy = torch.empty((molecule_num), dtype=torch.float)
    
        for molecule_idx in range(molecule_num):
            comment, atoms = molecules[molecule_idx]
            for com in comment.split():
                if com.startswith("energy="):
                    energy[molecule_idx] = torch.tensor(float(com.split("=")[1]), dtype=torch.float)
            for atom_idx in range(atom_num):
                atom_type, px, py, pz, fx, fy, fz = atoms[atom_idx]
                z[molecule_idx, atom_idx] = torch.tensor(atom_type2z[atom_type], dtype=torch.long)
                pos[molecule_idx, atom_idx] = torch.tensor([px, py, pz])
                force[molecule_idx, atom_idx] = torch.tensor([fx, fy, fz])

        self.z, self.pos, self.force, self.energy = z, pos, force, energy
        
        # import numpy as np
        
        # it = zip(self.raw_paths, self.processed_paths)
        
        # for raw_path, processed_path in it:
        #     raw_data = np.load(raw_path)

        #     data_list = []
        #     import tqdm
        #     for i in tqdm.tqdm(range(pos.size(0))):
        #         data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
        #         if self.pre_filter is not None and not self.pre_filter(data):
        #             continue
        #         if self.pre_transform is not None:
        #             data = self.pre_transform(data)
        #         data_list.append(data)

        #     torch.save(self.collate(data_list), processed_path)
        
    def to(self, device):
        self.z = self.z.to(device)
        self.pos = self.pos.to(device)
        self.force = self.force.to(device)
        self.energy = self.energy.to(device)

    # def __getitem__(self, index):
    #     return self.data[index]
        
    def __len__(self):
        return self.energy.shape[0]

    def process(self):
        z, pos, force, energy = self.z, self.pos, self.force, self.energy
        for i in range(z.shape[1]):
            assert z[:, i].unique().shape[0] == 1
        z = z[0, :]
        data_list = []
        import tqdm
        for i in tqdm.tqdm(range(pos.size(0))):
            data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)


        torch.save(self.collate(data_list), self.processed_paths[0])
        return data_list
    
        
def triBPA_datawork(    
    root: str, 
    name: str,
    transform,
    max_neighbor,
    batch_size: List[int],
    subgraph_cutoff: float = -1.0,
    cutoff: float = -1.0,
    extend_r=1.0,
    **kwargs
    ):
    
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
    
    
    # get dataset and collate function.
    datasets = []
    for data_name in ["train_300K", "test_300K", "test_600K", "test_1200K"]:
        dataset = triBPA_dataset(root=root, name=data_name, transform=transform_)
        datasets.append(dataset)
    train_300K_dataset, test_300K_dataset, test_600K_dataset, test_1200K_dataset = datasets
    
    
    # get meta data
    # TODO: change to pure train data
    global_y_mean = torch.cat([train_300K_dataset.energy, test_300K_dataset.energy, test_600K_dataset.energy, test_1200K_dataset.energy]).mean()
    global_y_std = torch.cat([train_300K_dataset.energy, test_300K_dataset.energy, test_600K_dataset.energy, test_1200K_dataset.energy]).std()


    # get basic configs
    train_data_num, val_data_num = [450, 50]
    train_batch_size, val_batch_size, test_batch_size = batch_size
    
    # random_split dataset
    train_dataset, val_dataset = random_split(
        train_300K_dataset, 
        [train_data_num, 
        val_data_num]
        )

    
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader_300K, test_dataloader_600K, test_dataloader_1200K = (
        DataLoader(train_dataset, num_workers=4, batch_size=train_batch_size, persistent_workers=True, shuffle=True, collate_fn=collate_),
        DataLoader(val_dataset, num_workers=4, batch_size=val_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_),
        DataLoader(test_300K_dataset, num_workers=4, batch_size=test_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_),
        DataLoader(test_600K_dataset, num_workers=4, batch_size=test_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_),
        DataLoader(test_1200K_dataset, num_workers=4, batch_size=test_batch_size, persistent_workers=True, shuffle=False, collate_fn=collate_)
    )
    
    test_dataloader = [test_dataloader_300K, test_dataloader_600K, test_dataloader_1200K]
    

    return train_dataloader, val_dataloader, test_dataloader, global_y_mean, global_y_std
