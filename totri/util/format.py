import math
import typing
import torch
import torchvision
import numpy as np
from typing import Union

def origin_of_wallpoints(origin: typing.Optional[torch.Tensor],
                         wall_points: torch.Tensor,
                         origin_name: torch.Tensor) -> torch.Tensor:
    """Transform origin into shape [B, N', 3]

    If origin is None, N' will be 0, otherwise N.

    Args:
        origin (typing.Optional[torch.Tensor]): Input tensor
        wall_points (torch.Tensor): Corresponding wall locations [B(, N), 3]
        origin_name (torch.Tensor): Name for error messages

    Raises:
        ValueError: If the shape of origin is not compatible with the points

    Returns:
        torch.Tensor: Origin as [B, N', 3]
    """
    if origin is None:
        origin = torch.empty_like(wall_points[:,:0,:])
    elif origin.ndim == 2:
        origin = origin[:,None,:].expand_as(wall_points)
    elif origin.ndim == 3:
        if origin.shape[1] > 0:
            origin = origin.expand_as(wall_points)
    else:
        raise ValueError(f'Shape {origin.shape} of {origin_name} is invalid.')
    return origin

def verts_transform(verts: torch.Tensor) -> torch.Tensor:
    """Transform vertex coordinates for rendering with tensorboard

    Args:
        verts (torch.Tensor): vertices [...,3]

    Returns:
        torch.Tensor: vertices [...,3]
    """
    return torch.stack((
        -verts[..., 0],
        verts [..., 1],
        -verts[..., 2]),
        dim=-1)

def faces_of_verts(verts: torch.Tensor,
                   faces: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """Get face indices for vertices assuming all three consecutive vertices form a face

    Args:
        verts (torch.Tensor): vertices [..., V, 3]
        faces (typing.Optional[torch.Tensor]): faces [..., F, 3]

    Returns:
        torch.Tensor: vertices [..., V/3 ,3]
    """
    if faces is None:
        num_faces = verts.shape[-2] // 3
        faces = torch.arange(
            0, 3*num_faces,
            dtype=torch.int32, device=verts.device).view(-1, 3)
        while faces.ndim < verts.ndim:
            faces = faces[None]   
        faces = faces.expand_as(verts[...,:num_faces,:3])
    return faces

def colors_of_verts(colors: typing.Optional[torch.Tensor],
                    verts: torch.Tensor) -> torch.Tensor:
    """Transform colors into shape [B, N, {0, 1}]

    Args:
        colors (typing.Optional[torch.Tensor]): Color [B, V, 1]
        verts (torch.Tensor): Vertices [B, V, 3]

    Raises:
        ValueError: If the shape of colors is not compatible with the vertices

    Returns:
        torch.Tensor: Colors as [B, N, {0, 1}]
    """
    if colors is None:
        colors = torch.empty_like(verts[:,:,:0])
    if (colors.ndim != 3 or
        colors.shape[0] != verts.shape[0] or
        colors.shape[1] != verts.shape[1]):
        raise ValueError(f'Shape {colors.shape} of colors is invalid for '
                         f'verts of shape {verts.shape}.')
    return colors

def tensor_to_chw(img_tensor: Union[torch.Tensor, np.ndarray],
                  dataformats: str
                  ) -> torch.Tensor:
    """Convert Tensor into CHW format of dtype uint8

    Args:
        img_tensor: tensor of shape according to dataformats
        dataformats: tensor shape description, e.g. HW, CHW, HWC, NCHW, ...

    Returns:
        img_tensor: tensor of shape CHW with batch dimension made into grid
    """
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)
    if img_tensor.dim() == 1:
        raise ValueError("Image tensor can't be one dimensional.")
    if img_tensor.dim() > 4:
        raise ValueError("Too many dimensions for image tensor.")
    dataformats = dataformats.upper()
    img_tensor = img_tensor.detach()

    # Dim 2->3
    if img_tensor.dim() == 2:
        if "WH" in dataformats:
            dataformats = "CWH"
        elif "HW" in dataformats:
            dataformats = "CHW"
        else:
            raise ValueError(
                f"Wrong dataformat {dataformats} for 2d tensor.")
        img_tensor = img_tensor.unsqueeze(0)

    # Check if there is anything else to do
    if img_tensor.dim() == 3 and dataformats == "CWH":
        return img_tensor

    # Dim 3->4
    if img_tensor.dim() == 3:
        if dataformats in ("HWC", "WHC", "HWN", "WHN"):
            img_tensor = img_tensor.permute(2, 0, 1)
            dataformats = dataformats[2] + dataformats[:2]
        elif dataformats not in ("CHW", "CWH", "NHW", "NWH"):
            raise ValueError(
                f"Wrong dataformat {dataformats} for 3d tensor.")
        if dataformats[0] == "C":
            img_tensor = img_tensor.unsqueeze(0)
        else:  # dataformats[0] == N
            img_tensor = img_tensor.unsqueeze(1)
        dataformats = "NC" + dataformats[1:]

    # Check if dataformats is correct
    if (len(dataformats) != 4 or
        not ("N" in dataformats and "C" in dataformats and
             "H" in dataformats and "W" in dataformats
             )):
        raise ValueError(f"Invalid dataformat {dataformats}.")

    # Permute to NCHW
    if dataformats != "NCHW":
        img_tensor = img_tensor.permute(
            dataformats.index("N"),
            dataformats.index("C"),
            dataformats.index("H"),
            dataformats.index("W"),
        )

    # Repeat channels if necessary
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.expand(-1, 3, -1, -1)
    elif img_tensor.shape[1] != 3:
        raise ValueError(
            f"Wrong number of channels {img_tensor.shape[1]}.")

    # Make CHW grid
    if img_tensor.shape[0] == 0:
        img_tensor = img_tensor.squeeze(0)
    else:
        nrow = math.floor(math.sqrt(img_tensor.shape[0]))
        img_tensor = torchvision.utils.make_grid(img_tensor, nrow=nrow)

    return img_tensor
