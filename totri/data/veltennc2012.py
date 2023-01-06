import typing
import torch
import numpy as np
from scipy.io import loadmat

from totri.types import BinDef, VolumeDef

def _DefaultTransformVeltenNc2012(x: torch.Tensor, is_normal: bool) -> torch.Tensor:
    """Transform coordinates/normals

    The default transformation is a (-3.75, -2.20, 0) m translation followed by
    a 25 degree rotation.

    Args:
        x (torch.Tensor): coordinates/normals to be transformed [N, 3]
        is_normal (bool): whether x contains coordinates or normals

    Returns:
        torch.Tensor: transformed x [N, 3]
    """
    translation = torch.tensor(
        [[-3.75, -2.20, 0.0]],
        device=x.device,
        dtype=x.dtype)
    angle = 25/360*2*np.pi
    rotation = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle),  np.cos(angle), 0.0],
            [          0.0,            0.0, 1.0],
        ], 
        device=x.device,
        dtype=x.dtype)
    if not is_normal:
        x = x + translation
    x = torch.matmul(x, rotation.T)
    x = torch.stack((x[:,1], x[:,2], -x[:,0]), dim=1)
    if not is_normal:
        x = x + torch.tensor(
            [[0.0, 0.0, 2.0]],
            device=x.device,
            dtype=x.dtype)
    return x

class VeltenNc2012Dataset():
    __slots__ = ['transient', 'scan_points', 'laser_points',
                 'volume_def', 'bin_def',
                 'scan_origin', 'laser_origin']
    transient: torch.Tensor
    """Transient measurement volume [1, N, S, L]"""
    scan_points: torch.Tensor
    """Scan points on the visible wall [1, S, 3]"""
    laser_points: torch.Tensor
    """Laser points on the visible wall [1, L, 3]"""
    volume_def: VolumeDef
    """Definition of the volume"""
    bin_def: BinDef
    """Bin width and offset"""
    scan_origin: torch.Tensor
    """Position of the detector for each scan point [1, 3]"""
    laser_origin: torch.Tensor
    """Position of the laser for each laser point [1, 3]"""

    def rectify(self):
        """Shift transient bins to eliminate origin to wall distances"""
        if self.scan_origin is not None:
            for s in range(self.scan_points.shape[1]):
                distance = ((self.scan_points[:, s, :] - self.scan_origin)**2).sum()**0.5
                bin_shift = distance / self.bin_def.width
                self.transient[0, :, s, :] = torch.roll(
                    self.transient[0, :, s, :],
                    -bin_shift.int().item(),
                    0)
            self.scan_origin = None
        if self.laser_origin is not None:
            for l in range(self.laser_points.shape[1]):
                distance = ((self.laser_points[:, l, :] - self.laser_origin)**2).sum()**0.5
                bin_shift = distance / self.bin_def.width
                self.transient[0, :, :, l] = torch.roll(
                    self.transient[0, :, :, l],
                    -bin_shift.int().item(),
                    0)
            self.laser_origin = None

def VeltenNc2012(path: str, transform: typing.Callable = _DefaultTransformVeltenNc2012,
                 rectify: bool = False,
                 device: torch.device = torch.device('cpu')) -> VeltenNc2012Dataset:
    """Load data from a .mat file with the VeltenNc2012 format

    The .mat file must be structured as in:
    Velten, A., Willwacher, T., Gupta, O., Veeraraghavan, A., Bawendi, M. G., & Raskar, R. (2012).
    Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging.
    Nature communications, 3(1), 1-8.

    It must contain the following variables:
    laserOrigin, cameraOrigin, cameraPos, deltat, t0, data, t, laserPos, laserNorm, cameraNorm

    In the original files all measurements are given as cm. We convert it to m.

    Args:
        path (str): path to .mat file

    Returns:
        VeltenNc2012Dataset: Loaded dataset
    """
    mat = loadmat(path)
    data = VeltenNc2012Dataset()

    bin_width = mat['dataset']['deltat'][0,0].item() / 100 # cm to m
    bin_offset = mat['dataset']['t0'][0,0].item() / 100 # cm to m
    bin_num = mat['dataset']['t'][0,0].item()
    data.bin_def = BinDef(bin_width, bin_offset, bin_num)

    transient = mat['dataset']['data'][0,0] # [L, S*N]
    transient = transient.reshape(1, transient.shape[0], -1, bin_num) # [1, L, S, N]
    data.transient = torch.tensor(transient, dtype=torch.float32, device=device).permute(0, 3, 2, 1) # [1, N, S, L]

    scan_points = mat['dataset']['cameraPos'][0,0] / 100  # [S, 3]
    scan_points = transform(torch.tensor(scan_points, dtype=torch.float32, device=device), False) # [S, 3]
    data.scan_points = scan_points[None] # [1, S, 3]

    laser_points = mat['dataset']['laserPos'][0,0] / 100  # [L, 3]
    laser_points = transform(torch.tensor(laser_points, dtype=torch.float32, device=device), False) # [L, 3]
    data.laser_points = laser_points[None] # [1, L, 3]

    scan_origin = mat['dataset']['cameraOrigin'][0,0] / 100  # [1, 3]
    scan_origin = transform(torch.tensor(scan_origin, dtype=torch.float32, device=device), False) # [1, 3]
    data.scan_origin = scan_origin # [1, 3]

    laser_origin = mat['dataset']['laserOrigin'][0,0] / 100  # [1, 3]
    laser_origin = transform(torch.tensor(laser_origin, dtype=torch.float32, device=device), False) # [1, 3]
    data.laser_origin = laser_origin # [1, 3]

    data.volume_def = VolumeDef(
        (-0.75, 0.0, -0.44+2.0),
        ( 0.75, 3.0,  0.44+2.0),
        ( 44,   150,        75)
    )

    if rectify:
        data.rectify()

    return data
