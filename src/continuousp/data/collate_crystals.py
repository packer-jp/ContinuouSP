from continuousp.models.crystals import Crystals
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData


def collate_crystals(batch: list[BaseData]) -> Crystals:
    batch = Batch.from_data_list(batch)
    return Crystals(
        batch.cell,
        batch.pos,
        batch.atomic_numbers,
        batch.natoms,
        batch.batch,
    )
