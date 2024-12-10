from pathlib import Path

import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data

from continuousp.data.dataset_partition import DatasetPartition
from continuousp.utils.logger import LOGGER


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
            for row in pd.read_csv(path_csv).itertuples():
                crystal = Structure.from_str(
                    row.cif,
                    fmt='cif',
                )
                # if partition == DatasetPartition.TEST:
                #    crystal = crystal.get_primitive_structure()
                crystal = crystal.get_reduced_structure()
                data = Data(
                    cell=torch.FloatTensor(crystal.lattice.matrix.copy()).view(1, 3, 3),
                    pos=torch.FloatTensor(crystal.cart_coords),
                    atomic_numbers=torch.LongTensor(crystal.atomic_numbers),
                    natoms=torch.LongTensor([crystal.num_sites]),
                    composition_id=str(row.material_id),
                    formation_energy_per_atom=torch.FloatTensor(
                        [
                            [
                                row.formation_energy_per_atom
                                if hasattr(row, 'formation_energy_per_atom')
                                else row.heat_ref
                                if hasattr(row, 'heat_ref')
                                else row.energy_per_atom + 154,
                            ],
                        ],
                    ),
                )
                self.data_list.append(data)
            torch.save(self.data_list, path_pth)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Data:
        return self.data_list[index]
