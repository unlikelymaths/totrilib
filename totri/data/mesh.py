from json import load
import os
import typing
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes

def disable_logging(modules):
    """Set logging level of given modules to CRITICAL

    Args:
        modules (list[str]): List of modules
    """
    import logging
    for module in modules:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

OBJS = {
    'armadillo': 'armadillo.obj',
    'bunny': 'bunny.obj',
    'mannequin': 'mannequin.obj',
    'mouse0': 'mouse0.obj',
    'mouse1': 'mouse1.obj',
}

COLOR_OBJS = {
    'spot': 'spot/cow.obj',
    'spot_hi': 'spot/cow_hi.obj',
}

def load_sample_obj(id: str, device: torch.device=torch.device('cuda')
        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Load sample obj as a mesh

    Supported ids:
    - bunny

    Args:
        id (str): Sample id
        device (torch.device): device to load mesh (cpu or cuda)

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: (verts, faces) of the mesh [1, N, 3], [1, F, 3]
    """
    # iopath event logger keeps throwing errors we do not care about
    disable_logging(['iopath.common.file_io'])
    obj_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../data/obj',
        OBJS[id])
    verts, faces, aux = load_obj(obj_path)
    faces = faces.verts_idx
    return verts[None].to(device=device), faces[None].to(device=device)

def load_sample_obj_color(id: str, device: torch.device=torch.device('cuda')
        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Load sample obj as a mesh

    Supported ids:
    - spot

    Args:
        id (str): Sample id
        device (torch.device): device to load mesh (cpu or cuda)

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: (verts, colors, faces) of the mesh [1, N, 3], [1, N, 3], [1, F, 3]
    """
    # iopath event logger keeps throwing errors we do not care about
    disable_logging(['iopath.common.file_io'])
    obj_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../data/obj',
        COLOR_OBJS[id])
    mesh = load_objs_as_meshes([obj_path,], device='cpu')
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    faces_colors = mesh.textures.faces_verts_textures_packed()
    colors = torch.zeros_like(verts)
    for f in range(faces.shape[0]):
        colors[faces[f, 0], :] = faces_colors[f, 0, :]
        colors[faces[f, 1], :] = faces_colors[f, 1, :]
        colors[faces[f, 2], :] = faces_colors[f, 2, :]
    if id == 'spot' or id == 'spot_hi':
        verts = torch.stack((-verts[:,2]+0.2, verts[:,1], verts[:,0]+1.5), dim=-1)
        
    return verts[None].to(device=device), colors[None].to(device=device), faces[None].to(device=device)

def load_sample_mesh(id: str, device: torch.device=torch.device('cuda')
        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """[summary]

    Args:
        id (str): Sample id
        device (torch.device): device to load mesh (cpu or cuda)

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (verts, faces) of the mesh [1, N, 3], [1, F, 3]
            (verts, colors, faces) of the mesh [1, N, 3], [1, N, 3], [1, F, 3]
    """
    if id in OBJS:
        return load_sample_obj(id, device)
    elif id in COLOR_OBJS:
        return load_sample_obj_color(id, device)
    else:
        raise KeyError(f'Sample "{id}" does not exist.')
