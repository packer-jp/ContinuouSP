import math
from collections.abc import Generator

import numpy as np
import torch
import torch.nn.functional
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import (
    global_mean_pool,
)

from continuousp.models.continuous_g_conv import ContinuousGConv
from continuousp.models.crystals import Crystals
from continuousp.models.gaussian_smearing import GaussianSmearing
from continuousp.utils.logger import LOGGER


class ContinuousEnergyPredictor(torch.nn.Module):
    def __init__(
        self,
        num_atomic_numbers: int,
        atom_embedding_size: int,
        edge_feat_size: int,
        num_graph_conv_layers: int,
        fc_feat_size: int,
        num_fc_layers: int,
        num_gen_steps: int,
        gaussian_stop: float,
        basis_width_scalar: float,
        radius_rate: float,
        step_size_start: float,
        step_size_end: float,
        temp_start: float,
        temp_end: float,
        expected_density: float,
        energy_penalty: float,
    ) -> None:
        super().__init__()
        self.num_atomic_numbers = num_atomic_numbers
        self.radius_rate = radius_rate
        self.atom_embedding_size = atom_embedding_size
        self.edge_feat_size = edge_feat_size
        self.num_graph_conv_layers = num_graph_conv_layers
        self.fc_feat_size = fc_feat_size
        self.num_fc_layers = num_fc_layers
        self.expected_density = expected_density
        self.energy_penalty = energy_penalty

        self.step_size_schedule = np.logspace(
            math.log10(step_size_start),
            math.log10(step_size_end),
            num=num_gen_steps,
        )
        self.temp_schedule = np.logspace(
            math.log10(temp_start),
            math.log10(temp_end),
            num=num_gen_steps,
        )

        self.embedding = torch.nn.Embedding(num_atomic_numbers, atom_embedding_size)
        self.distance_expansion = GaussianSmearing(
            0.0,
            gaussian_stop,
            edge_feat_size,
            basis_width_scalar,
        )
        self.conv_layers = torch.nn.ModuleList(
            [
                ContinuousGConv(
                    atom_embedding_size,
                    edge_feat_size,
                )
                for _ in range(num_graph_conv_layers)
            ],
        )

        self.fc_layers = torch.nn.ModuleList(
            [
                spectral_norm(
                    torch.nn.Linear(
                        fc_feat_size if i > 0 else atom_embedding_size,
                        fc_feat_size,
                    ),
                )
                for i in range(num_fc_layers - 1)
            ],
        )

        self.fc_final = torch.nn.Linear(
            fc_feat_size,
            1,
        )

    def _construct_graph(  # noqa: PLR0915
        self,
        crystal_batch: Crystals,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the crystal graph.

        MIT License

        Copyright (c) Meta, Inc. and its affiliates.

        Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        """  # noqa: E501
        num_atoms_per_image = crystal_batch.natoms
        num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

        index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

        index_offset_expand = torch.repeat_interleave(
            index_offset,
            num_atoms_per_image_sqr,
        )
        num_atoms_per_image_expand = torch.repeat_interleave(
            num_atoms_per_image,
            num_atoms_per_image_sqr,
        )

        num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
        index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
        index_sqr_offset = torch.repeat_interleave(
            index_sqr_offset,
            num_atoms_per_image_sqr,
        )
        atom_count_sqr = (
            torch.arange(num_atom_pairs, device=crystal_batch.device) - index_sqr_offset
        )

        index1 = (
            torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode='floor')
        ) + index_offset_expand
        index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand

        pos1 = torch.index_select(crystal_batch.pos, 0, index1)
        pos2 = torch.index_select(crystal_batch.pos, 0, index2)

        cross_a2a3 = torch.cross(
            crystal_batch.cell[:, 1],
            crystal_batch.cell[:, 2],
            dim=-1,
        )
        cell_vol = torch.sum(
            crystal_batch.cell[:, 0] * cross_a2a3,
            dim=-1,
            keepdim=True,
        )

        radius = self.radius_rate * (torch.abs(cell_vol.view(-1)) / crystal_batch.natoms) ** (1 / 3)

        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)

        cross_a3a1 = torch.cross(
            crystal_batch.cell[:, 2],
            crystal_batch.cell[:, 0],
            dim=-1,
        )
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)

        cross_a1a2 = torch.cross(
            crystal_batch.cell[:, 0],
            crystal_batch.cell[:, 1],
            dim=-1,
        )
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)

        max_rep = [rep_a1.max().item(), rep_a2.max().item(), rep_a3.max().item()]

        cells_per_dim = [
            torch.arange(-rep, rep + 1, device=crystal_batch.device, dtype=torch.float)
            for rep in max_rep
        ]
        unit_cell = torch.cartesian_prod(*cells_per_dim)
        num_cells = len(unit_cell)
        unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
        unit_cell = torch.transpose(unit_cell, 0, 1)
        unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
            crystal_batch.batch_size,
            -1,
            -1,
        )

        data_cell = torch.transpose(crystal_batch.cell, 1, 2)
        pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
        pbc_offsets_per_atom = torch.repeat_interleave(
            pbc_offsets,
            num_atoms_per_image_sqr,
            dim=0,
        )

        pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
        pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
        index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
        index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)

        pos2 = pos2 + pbc_offsets_per_atom

        atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
        atom_distance_sqr = atom_distance_sqr.view(-1)

        radius = torch.index_select(
            radius,
            0,
            torch.index_select(crystal_batch.batch, 0, index1),
        ).view(-1)

        mask_within_radius = torch.le(atom_distance_sqr, radius * radius)

        mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
        mask = torch.logical_and(mask_within_radius, mask_not_same)
        index1 = torch.masked_select(index1, mask)
        index2 = torch.masked_select(index2, mask)
        unit_cell = torch.masked_select(
            unit_cell_per_atom.view(-1, 3),
            mask.view(-1, 1).expand(-1, 3),
        )
        unit_cell = unit_cell.view(-1, 3)
        atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
        atom_distance = torch.sqrt(atom_distance_sqr)
        radius = torch.masked_select(radius, mask)

        edge_index = torch.stack((index2, index1))

        return edge_index, atom_distance, radius

    def forward(self, data: Crystals) -> torch.Tensor:
        (
            edge_index,
            distance,
            radius,
        ) = self._construct_graph(data)
        feat = torch.nn.functional.tanh(self.embedding(data.atomic_numbers - 1))
        edge_weight = torch.cos(distance * math.pi / radius).view(-1, 1) + 1
        edge_attr = self.distance_expansion(distance)
        for conv_layer in self.conv_layers:
            feat = conv_layer(feat, edge_index, edge_weight, edge_attr)
            feat = torch.nn.functional.softplus(feat)
        feat = global_mean_pool(feat, data.batch)
        for fc_layer in self.fc_layers:
            feat = fc_layer(feat)
            feat = torch.nn.functional.softplus(feat)
        result = self.fc_final(feat)
        density = (data.natoms / torch.abs(torch.linalg.det(data.cell))).view(-1, 1)
        return result + self.energy_penalty * (torch.log(density / self.expected_density)) ** 2

    def generate(
        self,
        atomic_numbers: torch.Tensor,
        natoms: torch.Tensor,
        composition_id: list[str],
    ) -> Generator[Crystals, None, None]:
        current_crystals = (
            Crystals.create_randomly(
                atomic_numbers,
                natoms,
                composition_id,
                self.expected_density,
            )
            .reduce()
            .require_grad()
        )

        self.zero_grad()
        current_energy = self.forward(current_crystals)

        current_energy.sum().backward()
        yield current_crystals.detach()

        for step, (step_size, temp) in enumerate(
            zip(self.step_size_schedule, self.temp_schedule, strict=False),
        ):
            LOGGER.debug({'Step': step})
            with torch.cuda.nvtx.range('transition'):
                transitioned_crystals = (
                    current_crystals.transition(step_size, temp).detach().require_grad()
                )
            with torch.cuda.nvtx.range('reduce'):
                transitioned_crystals_reduced = transitioned_crystals.reduce().retain_grad()
            with torch.cuda.nvtx.range('forward'):
                self.zero_grad()
                transitioned_energy = self.forward(transitioned_crystals_reduced)
            with torch.cuda.nvtx.range('backward'):
                transitioned_energy.sum().backward()
            with torch.cuda.nvtx.range('rest'):
                log_accept_ratio = (
                    (current_energy - transitioned_energy) / temp
                    + current_crystals.log_transition_prob(
                        transitioned_crystals,
                        step_size,
                        temp,
                    )
                    - transitioned_crystals.log_transition_prob(
                        current_crystals,
                        step_size,
                        temp,
                    )
                )
                update_mask = torch.log(torch.rand_like(log_accept_ratio)) < log_accept_ratio
                current_crystals = Crystals.where(
                    update_mask,
                    transitioned_crystals_reduced,
                    current_crystals,
                )
                current_energy = torch.where(
                    update_mask,
                    transitioned_energy,
                    current_energy,
                )
                yield current_crystals.detach()
