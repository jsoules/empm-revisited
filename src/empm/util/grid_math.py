from torch import linspace, float32, Tensor


def generate_equispaced_points(half_radius: float, n_points: int) -> Tensor:
    return linspace(-half_radius, half_radius, n_points).to(dtype=float32)
