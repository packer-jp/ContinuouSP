from pathlib import Path

import pandas as pd
import torch
from continuousp.data.dataset_partition import DatasetPartition
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data


class CrystalDataset(Dataset):
    def __init__(self, dataset: str, partition: DatasetPartition) -> None:
        super().__init__()
        rootname = Path('./data') / dataset / partition.value
        path_pth = rootname.with_suffix('.pth')
        path_csv = rootname.with_suffix('.csv')

        if path_pth.exists():
            self.data_list = torch.load(path_pth)
        else:
            self.data_list = []
            for index, row in pd.read_csv(path_csv).iterrows():
                crystal = Structure.from_str(
                    row['cif'],
                    fmt='cif',
                ).get_reduced_structure()
                data = Data(
                    id=torch.LongTensor([index]),
                    cell=torch.LongTensor(crystal.atomic_numbers),
                    pos=torch.FloatTensor(crystal.cart_coords),
                    atomic_numbers=torch.FloatTensor(crystal.lattice.matrix).view(1, 3, 3),
                    natoms=torch.LongTensor([crystal.num_sites]),
                )
                self.data_list.append(data)
            torch.save(self.data_list, path_pth)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Data:
        return self.data_list[index]
