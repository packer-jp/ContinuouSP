import torch


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float,
        stop: float,
        num_gaussians: int,
        basis_width_scalar: float,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1)) ** 2)
