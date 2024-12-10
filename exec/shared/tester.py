from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from shared.should_save import should_save
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

import wandb
from continuousp.data.collate_crystals import collate_crystals
from continuousp.data.crystal_dataset import CrystalDataset
from continuousp.data.dataset_partition import DatasetPartition
from continuousp.models.continuous_energy_predictor import ContinuousEnergyPredictor
from continuousp.models.crystals import Crystals
from continuousp.utils.logger import LOGGER
from continuousp.vis.show_crystal_timeline import show_single_structure


class Tester:
    def __init__(
        self,
        run_name: str | None,
        train_run_id: str,
        config: dict,
        num_test_data: int,
        device: torch.device,
    ) -> None:
        self.model = ContinuousEnergyPredictor(**config['model']).to(device)
        self.model.eval()

        test_dataset = CrystalDataset(
            dataset=config['training']['dataset'],
            partition=DatasetPartition.TRAIN,
        )
        if num_test_data != -1:
            test_dataset = Subset(test_dataset, range(num_test_data))

        self.test_data_loader = DataLoader(
            test_dataset,
            batch_size=config['testing']['batch_size'],
            shuffle=False,
            collate_fn=collate_crystals,
        )

        self.run = wandb.init(
            project='ContinuouSP-test',
            name=run_name,
            config=config,
            mode='disabled' if run_name is None else 'online',
        )
        self.run.log_code(
            root=str(Path(__file__).parent.parent.parent),
            include_fn=should_save,
        )

        self.snapshot_path = (Path('./snapshots') / train_run_id).with_suffix('.pth')
        self._load_snapshot()

        self.device = device

    def __del__(self) -> None:
        self.run.finish()

    def _load_snapshot(self) -> None:
        LOGGER.info('Loading snapshot')
        snapshot = torch.load(self.snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])

    def get_rms_dists(self) -> Generator[float | None, None, None]:
        for known_crystals in self.test_data_loader:
            known_crystals = known_crystals.to(self.device)
            known_pymatgen_structures = known_crystals.to_pymatgen_structures()
            generated_pymatgen_structures = list(
                self.model.generate(
                    known_crystals.atomic_numbers,
                    known_crystals.natoms,
                    known_crystals.composition_id,
                ),
            )[-1].to_pymatgen_structures()
            matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
            for known_pymatgen_structure, generated_pymatgen_structure in zip(
                known_pymatgen_structures,
                generated_pymatgen_structures,
                strict=True,
            ):
                rms_dist = matcher.get_rms_dist(
                    known_pymatgen_structure,
                    generated_pymatgen_structure,
                )
                yield None if rms_dist is None else rms_dist[0]

    def step_vs_energy_and_structure(self) -> None:
        for known_crystals in self.test_data_loader:
            known_crystals = known_crystals.to(self.device)
            known_pymatgen_structures = known_crystals.to_pymatgen_structures(
                self.model.energy_with_penalty(known_crystals),
            )
            for generated_crystals in self.model.generate(
                known_crystals.atomic_numbers,
                known_crystals.natoms,
                known_crystals.composition_id,
            ):
                generated_pymatgen_structures = generated_crystals.to_pymatgen_structures(
                    self.model.energy_with_penalty(generated_crystals),
                )
                self.run.log(
                    {
                        f'{known_pymatgen_structure.properties["composition_id"]} / Energy / Known': known_pymatgen_structure.properties[  # noqa: E501
                            'energy'
                        ]
                        for known_pymatgen_structure in known_pymatgen_structures
                    }
                    | {
                        f'{known_pymatgen_structure.properties["composition_id"]} / Structure / Known': show_single_structure(  # noqa: E501
                            known_pymatgen_structure,
                        )
                        for known_pymatgen_structure in known_pymatgen_structures
                    }
                    | {
                        f'{generated_pymatgen_structure.properties["composition_id"]} / Energy / Generated': generated_pymatgen_structure.properties[  # noqa: E501
                            'energy'
                        ]
                        for generated_pymatgen_structure in generated_pymatgen_structures
                    }
                    | {
                        f'{generated_pymatgen_structure.properties["composition_id"]} / Structure / Generated': show_single_structure(  # noqa: E501
                            generated_pymatgen_structure,
                        )
                        for generated_pymatgen_structure in generated_pymatgen_structures
                    },
                )

    def magnification_vs_energy(self) -> None:
        self.run.define_metric('Magnification')
        for known_crystals in self.test_data_loader:
            known_crystals = known_crystals.to(self.device)
            for composition_id in known_crystals.composition_id:
                self.run.define_metric(
                    f'{composition_id} / Energy / Magnified',
                    step_metric='Magnification',
                )
            for magnification in np.linspace(-0.5, 0.5, 100):
                magnified_crystals = Crystals(
                    known_crystals.cell * 10**magnification,
                    known_crystals.pos * 10**magnification,
                    known_crystals.atomic_numbers,
                    known_crystals.natoms,
                    known_crystals.batch,
                    known_crystals.composition_id,
                    None,
                ).to(self.device)
                for composition_id, energy in zip(
                    magnified_crystals.composition_id,
                    self.model.energy_with_penalty(magnified_crystals),
                    strict=True,
                ):
                    self.run.log(
                        {
                            f'{composition_id} / Energy / Magnified': energy,
                            'Magnification': magnification,
                        },
                    )

    def distance_vs_energy(self) -> None:
        for known_crystals in self.test_data_loader:
            known_crystals = known_crystals.to(self.device)

            for atom_idx in range(known_crystals.pos.size(0)):
                distances: list[float] = []
                energies: list[float] = []

                for _ in range(1000):
                    perturbed_pos = known_crystals.pos.clone()
                    perturbed_pos[atom_idx] += 0.05 * torch.randn_like(
                        perturbed_pos[atom_idx],
                    )

                    perturbed_crystals = Crystals(
                        known_crystals.cell,
                        perturbed_pos,
                        known_crystals.atomic_numbers,
                        known_crystals.natoms,
                        known_crystals.batch,
                        known_crystals.composition_id,
                        None,
                    ).to(self.device)

                    for i, (composition_id, energy) in enumerate(
                        zip(
                            perturbed_crystals.composition_id,
                            self.model.energy_with_penalty(perturbed_crystals),
                            strict=True,
                        ),
                    ):
                        distances.append(
                            (perturbed_pos[atom_idx] - known_crystals.pos[atom_idx])
                            .norm(p=2)
                            .item(),
                        )
                        energies.append(energy.item())

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=distances,
                        y=energies,
                        mode='markers',
                        marker={'size': 6, 'color': 'blue', 'opacity': 0.7},
                        name=f'Atom {atom_idx}',
                    ),
                )
                fig.update_layout(
                    title=f'Distance vs Energy for Atom {atom_idx}',
                    xaxis_title='Distance',
                    yaxis_title='Energy',
                    template='plotly_white',
                )

                wandb.log({f'distance_vs_energy_atom_{atom_idx}': fig})
