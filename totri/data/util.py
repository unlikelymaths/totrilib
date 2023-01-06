"""totri.data.util"""
import os
from numpy import cumsum
import torch

def make_wall_grid(x_from=-1, x_to=1, x_res=32,
                   y_from=-1, y_to=1, y_res=32,
                   z_val=0, device=torch.device('cuda'),
                   x_fastest=True, flattened=True) -> torch.Tensor:
    """Flattened coordinate tensor

    Args:
        x_from: First x coordinate
        x_to: Last x coordinate
        x_res: Resolution in x
        y_from: First y coordinate
        y_to: Last y coordinate
        y_res: Resolution in y
        z_val: Depth of the grid
        device: Device of the returned tensor
        x_fastest: If True x coordinate will increase fastest

    Returns:
        torch.Tensor: wall coordinates [y_res*x_res, 3] or [y_res, x_res, 3]
    """
    x = torch.linspace(x_from, x_to, x_res, device=device)
    y = torch.linspace(y_from, y_to, y_res, device=device)
    if x_fastest:
        y, x = torch.meshgrid(y, x)
    else:
        x, y = torch.meshgrid(x, y)
    z = torch.full_like(x, fill_value=z_val)
    scan_points = torch.stack((x, y, z), dim=2)
    if flattened:
        return scan_points.view(-1, 3)
    else:
        return scan_points

def repo_path(relative_path: str) -> str:
    """Transform relative path inside the repository to an absolute path

    Args:
        relative_path (str): path relative to totrilib directory

    Returns:
        str: absolute path
    """
    base_path = os.path.join(os.path.dirname(__file__), '../..')
    return os.path.abspath(os.path.join(base_path, relative_path))

def transient_noise(transient: torch.Tensor,
                    scale: float = 1.0, bias: float = 0.5) -> torch.Tensor:
    """Add noise to the transient image

    Args:
        transient (torch.Tensor): Transient
        scale (float, optional): _description_. Defaults to 1.0.
        bias (float, optional): _description_. Defaults to 1.0.

    Returns:
        torch.Tensor: Noisy transient
    """
    std = transient.std() + 1.e-6
    rate = scale * (transient / std + bias * torch.rand_like(transient))
    return torch.poisson(rate) / scale * std

def merge_meshes(verts_list, faces_list):
    # [V, 3]
    # [F, 3]
    num_verts = cumsum([verts.shape[-2] for verts in verts_list])
    faces_list = [faces_list[0],] + [
        faces_list[i] + num_verts[i-1]
        for i in range(1, len(faces_list))]
    verts = torch.cat(verts_list, dim=-2)
    faces = torch.cat(faces_list, dim=-2)
    return verts, faces
