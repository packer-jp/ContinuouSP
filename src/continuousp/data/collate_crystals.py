from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from continuousp.models.crystals import Crystals


def collate_crystals(batch: list[BaseData]) -> Crystals:
    batch = Batch.from_data_list(batch)
    return Crystals(
        batch.cell,
        batch.pos,
        batch.atomic_numbers,
        batch.natoms,
        batch.batch,
        batch.composition_id,
        batch.formation_energy_per_atom,
    )
