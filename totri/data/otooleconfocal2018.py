import typing
import torch
import numpy as np
from scipy.io import loadmat

from totri.types import BinDef, VolumeDef
from totri.data.util import make_wall_grid

class OTooleConfocal2018Dataset():
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
        self.bin_def = BinDef(4e-12, 100*4e-12, 2048, unit_is_time=True)
        self.volume_def = VolumeDef([-0.4, -0.4, 0.2], [0.4, 0.4, 1.0], [512, 64, 64])
        self.scan_point_extent = (2, 2)

    def downsample_temporal(self, factor):
        if not isinstance(factor, int):
            raise ValueError('Downsample factor must be integer.')
        if factor < 1:
            raise ValueError('Downsample factor must be positive.')
        if factor == 1:
            return
        if self.transient.shape[1] % factor:
            raise ValueError('Number of bins must be divisible by factor.')
        transient_new = torch.zeros_like(self.transient[:,::factor])
        for i in range(factor):
            transient_new += self.transient[:,i::factor]
        self.transient = transient_new
        self.bin_def.width = self.bin_def.width * factor
        self.bin_def.num = self.bin_def.num // factor

    def downsample_spatial(self, resolution):
        self.transient = torch.nn.functional.interpolate(
            self.transient,
            size=resolution,
            mode="bilinear",
            align_corners=False)
        self.scan_points = torch.nn.functional.interpolate(
            self.scan_points.permute(0, 3, 1, 2),
            size=resolution,
            mode="bilinear",
            align_corners=False).permute(0, 2, 3, 1)

def OTooleConfocal2018(path: str,
                       dtype: torch.dtype = torch.float32,
                       device: torch.device = torch.device('cpu')) -> OTooleConfocal2018Dataset:
    ds = OTooleConfocal2018Dataset()
    # Transient
    mat = loadmat(path)
    transient = torch.tensor(mat['rect_data'].astype(np.float32)).permute(2, 1, 0) # [N, H, W]
    transient = transient.flip([1,])
    transient[:600] = 0
    ds.transient = transient[None].to(dtype=dtype, device=device)
    # Scan points
    ds.scan_points = make_wall_grid(
        ds.volume_def.start[0], ds.volume_def.end[0], 64,
        ds.volume_def.start[1], ds.volume_def.end[1], 64,
        z_val=0, flattened=False)[None].to(dtype=dtype, device=device)
    return ds
