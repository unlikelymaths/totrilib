import typing
import h5py
import torch

from totri.types import BinDef, VolumeDef
from totri.data.util import make_wall_grid

class LindellFk2019Dataset():
    __slots__ = ['is_confocal', 'bin_def', 'volume_def', 'scan_point_extent',
                 'transient', 'scan_points']
    is_confocal: bool
    """Confocal or exhaustive scan"""
    bin_def: BinDef
    """Bin width and offset"""
    volume_def: VolumeDef
    """Definition of the volume"""
    scan_point_extent: typing.Tuple[int]
    """Size of the scan point grid"""
    transient: torch.Tensor
    """Transient measurement volume [1, N, SH, SW]"""
    scan_points: torch.Tensor
    """Scan points on the visible wall [1, SH, SW, 3]"""

    def __init__(self):
        self.is_confocal = True
        self.bin_def = BinDef(32e-12, 0, 512, unit_is_time=True)
        self.volume_def = VolumeDef([-1, -1, 0], [1, 1, 2], [512, 512, 512])
        self.scan_point_extent = (2, 2)

    def resize(self, height, width):
        pass

def LindellFk2019(meas_mat_path, tof_mat_path, crop=512,
                  dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device('cpu')):
    # Load Files
    with h5py.File(tof_mat_path, "r") as f:
        tofgrid = torch.tensor(f.get("tofgrid")) # [W, H]
    with h5py.File(meas_mat_path, "r") as f:
        transient = torch.tensor(f.get("meas")) # [T, W, H]
    # Dataset
    ds = LindellFk2019Dataset()
    # Transient
    for x in range(transient.shape[1]):
        for y in range(transient.shape[2]):
            transient[:, x, y] = torch.roll(
                transient[:, x, y],
                (-torch.floor(tofgrid[x, y] / (ds.bin_def.width_s*1e12))).int().item())
    transient = transient[:crop]
    transient = transient.permute(0, 2, 1).flip([1,2])[None] # [1, T, H, W]
    ds.transient = transient.to(dtype=dtype, device=device)
    # Scan points
    ds.scan_points = make_wall_grid(
        -1, 1, 512,
        -1, 1, 512,
        z_val=0, flattened=False)[None].to(dtype=dtype, device=device)
    return ds
