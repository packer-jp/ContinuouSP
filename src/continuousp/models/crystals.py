import math
from dataclasses import dataclass
from functools import reduce
from typing import Self

import torch
from pymatgen.core.structure import Structure
from torch_niggli import niggli_reduce
from torch_scatter import scatter

from continuousp.utils.frac_cart_conversion import cart_to_frac, frac_to_cart


@dataclass
class Crystals:
    cell: torch.Tensor  # [batch_size, 3, 3]
    pos: torch.Tensor  # [num_atoms, 3]
    atomic_numbers: torch.Tensor  # [num_atoms]
    natoms: torch.Tensor  # [batch_size]
    batch: torch.Tensor  # [num_atoms]
    composition_id: list[str]

    @staticmethod
    def create_randomly(
        atomic_numbers: torch.Tensor,
        natoms: torch.Tensor,
        composition_id: list[str],
        expected_density: float,
    ) -> Self:
        device = atomic_numbers.device
        cell_coefficient = (
            natoms[:, None, None] / (2 * math.sqrt(2 / math.pi)) / expected_density
        ) ** (1 / 3)
        cell = cell_coefficient * torch.randn(
            size=(natoms.shape[0], 3, 3),
            device=device,
        )
        pos = frac_to_cart(
            natoms,
            torch.rand(size=(atomic_numbers.shape[0], 3), device=device),
            cell,
        )
        return Crystals(
            cell,
            pos,
            atomic_numbers,
            natoms,
            torch.repeat_interleave(
                torch.arange(len(natoms), device=cell.device, dtype=torch.long),
                natoms,
                dim=0,
            ),
            composition_id,
        )

    @staticmethod
    def where(
        condition: torch.Tensor,
        positive: Self,
        negative: Self,
    ) -> Self:
        assert positive.cell.requires_grad
        assert negative.cell.requires_grad
        assert positive.pos.requires_grad
        assert negative.pos.requires_grad
        assert positive.atomic_numbers is negative.atomic_numbers
        assert positive.natoms is negative.natoms
        assert positive.batch is negative.batch
        selected_cell = torch.where(condition[:, None], positive.cell, negative.cell)
        selected_pos = torch.where(condition[positive.batch], positive.pos, negative.pos)
        selected_cell.requires_grad_()
        selected_pos.requires_grad_()
        selected_cell.grad = torch.where(
            condition[:, None],
            positive.cell.grad,
            negative.cell.grad,
        )
        selected_pos.grad = torch.where(
            condition[positive.batch],
            positive.pos.grad,
            negative.pos.grad,
        )
        return Crystals(
            selected_cell,
            selected_pos,
            positive.atomic_numbers,
            positive.natoms,
            positive.batch,
            positive.composition_id,
        )

    @property
    def batch_size(self) -> int:
        return self.cell.size(0)

    @property
    def device(self) -> torch.device:
        return self.cell.device

    def to(self, device: torch.device) -> Self:
        return Crystals(
            self.cell.to(device),
            self.pos.to(device),
            self.atomic_numbers.to(device),
            self.natoms.to(device),
            self.batch.to(device),
            self.composition_id,
        )

    def detach(self) -> Self:
        detached_cell = self.cell.detach()
        detached_pos = self.pos.detach()
        return Crystals(
            detached_cell,
            detached_pos,
            self.atomic_numbers,
            self.natoms,
            self.batch,
            self.composition_id,
        )

    def reduce(self) -> Self:
        reduced_cell = torch.bmm(
            torch.bmm(
                niggli_reduce(self.cell, num_iterations=1000000),
                torch.inverse(self.cell),
            ).detach(),
            self.cell,
        )
        reduced_pos = frac_to_cart(
            self.natoms,
            cart_to_frac(self.natoms, self.pos, reduced_cell) % 1.0,
            reduced_cell,
        )
        return Crystals(
            reduced_cell,
            reduced_pos,
            self.atomic_numbers,
            self.natoms,
            self.batch,
            self.composition_id,
        )

    def transition(self, step_size: float, temp: float) -> Self:
        transitioned_cell = (
            self.cell
            - step_size * self.cell.grad / temp
            + math.sqrt(2 * step_size) * torch.randn_like(self.cell)
        )
        transitioned_pos = (
            self.pos
            - step_size * self.pos.grad / temp
            + math.sqrt(2 * step_size) * torch.randn_like(self.pos)
        )
        return Crystals(
            transitioned_cell,
            transitioned_pos,
            self.atomic_numbers,
            self.natoms,
            self.batch,
            self.composition_id,
        )

    def require_grad(self) -> Self:
        self.cell.requires_grad_()
        self.pos.requires_grad_()
        return self

    def retain_grad(self) -> Self:
        self.cell.retain_grad()
        self.pos.retain_grad()
        return self

    def log_transition_prob(
        self,
        to: Self,
        step_size: float,
        temp: float,
    ) -> torch.Tensor:
        assert self.atomic_numbers is to.atomic_numbers
        assert self.natoms is to.natoms
        assert self.batch is to.batch
        return -(
            scatter(
                ((self.pos - to.pos - step_size * to.pos.grad / temp) ** 2).sum(dim=1),
                self.batch,
                reduce='sum',
            )
            + ((self.cell - to.cell - step_size * to.cell.grad / temp) ** 2).sum(
                dim=[1, 2],
            )
        ).view(-1, 1) / (4 * step_size)

    def geometrically_multiplied_composition(
        self,
        p: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_atomic_number = self.atomic_numbers.max().item()
        count = torch.zeros(
            self.batch_size,
            max_atomic_number + 1,
            device=self.device,
            dtype=torch.long,
        ).scatter_add(
            0,
            self.batch.view(-1, 1).expand(-1, max_atomic_number + 1),
            self.atomic_numbers.view(-1, 1)
            .expand(-1, max_atomic_number + 1)
            .eq(torch.arange(max_atomic_number + 1, device=self.device).view(1, -1))
            .long(),
        )
        natoms = torch.sum(count, dim=1)
        count //= reduce(torch.gcd, count.t()).view(-1, 1) * (
            torch.distributions.Geometric(p)
            .sample([self.batch_size, 1])
            .to(self.device)
            .expand(-1, max_atomic_number + 1)
            .long()
            + 1
        )

        batch_unique, atom_numbers_unique = torch.where(count > 0)
        atomic_numbers = torch.repeat_interleave(
            atom_numbers_unique,
            count[batch_unique, atom_numbers_unique],
        )
        natoms = torch.sum(count, dim=1)
        return atomic_numbers, natoms

    def to_pymatgen_structures(self) -> list[Structure]:
        return [
            Structure(
                self.cell[i].detach().cpu().numpy(),
                self.atomic_numbers[self.batch == i].detach().cpu().numpy(),
                self.pos[self.batch == i].detach().cpu().numpy(),
                coords_are_cartesian=True,
                properties={'id': self.composition_id[i]},
            )
            for i in range(self.batch_size)
        ]
