import torch


def frac_to_cart(
    natoms: torch.Tensor,
    frac_pos: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    lattice_nodes = torch.repeat_interleave(cell, natoms, dim=0)
    return torch.einsum('bi,bij->bj', frac_pos.float(), lattice_nodes.float())


def cart_to_frac(
    natoms: torch.Tensor,
    cart_pos: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    inv_lattice = torch.pinverse(cell, rcond=1e-5)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, natoms, dim=0)
    return torch.einsum('bi,bij->bj', cart_pos.float(), inv_lattice_nodes.float())
