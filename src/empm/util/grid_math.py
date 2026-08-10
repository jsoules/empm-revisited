from torch import linspace, float32, Tensor


def generate_equispaced_points(half_radius: float, n_points: int, center: bool = True) -> Tensor:
    if center:
        return linspace(-half_radius, half_radius, n_points).to(dtype=float32)
    return (linspace(-half_radius, half_radius, n_points + 1)[:-1]).to(dtype=float32)
