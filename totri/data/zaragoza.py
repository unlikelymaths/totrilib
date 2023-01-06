import typing
import torch
import h5py
import numpy as np

from totri.types import BinDef, VolumeDef, C0

class ZaragozaDataset():
    __slots__ = ['is_confocal', 'bin_def', 'volume_def',
                 'scan_point_extent', 'laser_point_extent',
                 'transient', 'scan_points', 'laser_points',
                 'scan_normals', 'laser_normals', 'scan_origin', 'laser_origin']
    is_confocal: bool
    """Confocal or exhaustive scan"""
    bin_def: BinDef
    """Bin width and offset"""
    volume_def: VolumeDef
    """Definition of the volume"""
    scan_point_extent: typing.Tuple[float]
    """Size of the scan point grid"""
    laser_point_extent: typing.Tuple[float]
    """Size of the laser point grid"""
    transient: torch.Tensor
    """Transient measurement volume [1, N, SH, SW(, LH, LW)]"""
    scan_points: torch.Tensor
    """Scan points on the visible wall [1, SH, SW, 3]"""
    laser_points: typing.Optional[torch.Tensor]
    """Laser points on the visible wall [1, LH, LW, 3] if not confocal"""
    scan_normals: torch.Tensor
    """Scan normals on the visible wall [1, SH, SW, 3]"""
    laser_normals: typing.Optional[torch.Tensor]
    """Laser normals on the visible wall [1, LH, LW, 3] if not confocal"""
    scan_origin: torch.Tensor
    """Position of the detector for each scan point [1, 3]"""
    laser_origin: typing.Optional[torch.Tensor]
    """Position of the laser for each laser point [1, 3] if not confocal"""

    def rectify(self):
        """Shift transient bins to eliminate origin to wall distance"""
        if self.is_confocal:
            if self.scan_origin is None:
                return
            for x in range(self.scan_points.shape[2]):
                for y in range(self.scan_points.shape[1]):
                    distance = ((self.scan_points[:, y, x, :] - self.scan_origin)**2).sum()**0.5
                    bin_shift = 2 * distance / self.bin_def.width
                    self.transient[0, :, y, x] = torch.roll(
                        self.transient[0, :, y, x],
                        -bin_shift.int().item())
            self.scan_origin = None

def swap_yz(tensor):
    return torch.stack(
        (tensor[...,0], tensor[...,2], -tensor[...,1]),
        dim=-1)

def Zaragoza(path: str, rectify: bool = False,
             dtype: torch.dtype = torch.float32,
             device: torch.device = torch.device('cpu')) -> ZaragozaDataset:
    ds = ZaragozaDataset()
    with h5py.File(path, 'r') as file:
        # is_confocal
        is_confocal = file['isConfocal'][()]
        ds.is_confocal = is_confocal
        # bin_Def
        bin_width = file['deltaT'][()]
        bin_offset = file['t0'][()]
        bin_num = file['t'][()].item()
        ds.bin_def = BinDef(bin_width, bin_offset, bin_num, unit_is_time=False)
        # volume_def
        volume_position = swap_yz(torch.tensor(file['hiddenVolumePosition']).view(3)).tolist()
        volume_size = file['hiddenVolumeSize'][()]
        volume_start = (
            volume_position[0] - volume_size/2,
            volume_position[1] - volume_size/2,
            volume_position[2] - volume_size/2)
        volume_end = (
            volume_position[0] + volume_size/2,
            volume_position[1] + volume_size/2,
            volume_position[2] + volume_size/2)
        spacial_resolution = torch.tensor(file['cameraGridPoints']).view(2).int().tolist()
        volume_resolution = (bin_num, spacial_resolution[1], spacial_resolution[0])
        ds.volume_def = VolumeDef(volume_start, volume_end, volume_resolution)
        # scan_point_extent
        ds.scan_point_extent = torch.tensor(file['cameraGridSize']).view(2).float().tolist()
        # scan_point_extent
        if is_confocal:
            ds.laser_point_extent = None
        else:
            ds.laser_point_extent = torch.tensor(file['laserGridSize']).view(2).float().tolist()
        # transient
        transient = torch.tensor(file['data'])
        if is_confocal:
            ds.transient = transient.view(
                1, ds.bin_def.num, 6, spacial_resolution[1], spacial_resolution[0]
                ).sum(dim=2).to(dtype=dtype, device=device)
        else:
            raise NotImplementedError()
        # scan_points
        scan_points = torch.tensor(file['cameraGridPositions'])
        ds.scan_points = swap_yz(scan_points.view(1, 3, spacial_resolution[1], spacial_resolution[0]).permute(0, 2, 3, 1)).to(dtype=dtype, device=device)
        # laser_points
        if is_confocal:
            ds.laser_points = None
        else:
            laser_points = torch.tensor(file['laserGridPositions'])
            ds.laser_points = swap_yz(laser_points.view(1, 3, spacial_resolution[1], spacial_resolution[0]).permute(0, 2, 3, 1)).to(dtype=dtype, device=device)
        # scan_normals
        scan_normals = torch.tensor(file['cameraGridNormals'])
        ds.scan_normals = swap_yz(scan_normals.view(1, 3, spacial_resolution[1], spacial_resolution[0]).permute(0, 2, 3, 1)).to(dtype=dtype, device=device)
        # laser_normals
        if is_confocal:
            ds.laser_normals = None
        else:
            laser_normals = torch.tensor(file['laserGridNormals'])
            ds.laser_normals = swap_yz(laser_normals.view(1, 3, spacial_resolution[1], spacial_resolution[0]).permute(0, 2, 3, 1)).to(dtype=dtype, device=device)
        # scan_origin
        scan_origin = torch.tensor(file['cameraPosition'])
        ds.scan_origin = swap_yz(scan_origin.view(1, 3)).to(dtype=dtype, device=device)
        # laser_origin
        if is_confocal:
            ds.laser_origin = None
        else:
            laser_origin = torch.tensor(file['laserPosition'])
            ds.laser_origin = swap_yz(laser_origin.view(1, 3)).to(dtype=dtype, device=device)
    if rectify:
        ds.rectify()
    return ds
