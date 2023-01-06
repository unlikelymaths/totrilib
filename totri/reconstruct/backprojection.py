"""Fast Backprojection"""
import torch
import typing
from totri.types import VolumeDef, BinDef
from totri.util import origin_of_wallpoints
from totri_cpp.reconstruct import (
    backprojection_exhaustive_arellano_forward,
    backprojection_confocal_velten_forward,
    backprojection_exhaustive_velten_forward,
)

def second_derivative_filter(volume):
    def dz(volume):
        volume = torch.cat((volume, volume[:,-1:,:,:]), dim=1)
        return volume[:,1:,:,:]-volume[:,:-1,:,:]
    volume = dz(dz(volume))
    volume[volume>0] = 0
    return -volume

class BackprojectionExhaustiveArellano(torch.autograd.Function):
    """Compute probability volume by backprojecting transient points.

    Implementation follows a similar approach as the one proposed in:
    Arellano, V., Gutierrez, D., & Jarabo, A. (2017). Fast back-projection for non-line of sight
    reconstruction. Optics express, 25(10), 11574-11583.

    The scan and laser points and positions as well as the volume start and end points (x,y,z)
    correspond to (W,H,D). The wall is assumed to be at the xy plane for some
    fixed z. The volume resolution on the other hand is given as [D, H, W].

    Args:
        transient (torch.Tensor): Transient measurement volume [B, N, S, L]
        scan_points (torch.Tensor): Scan points on the visible wall [B, 3, S]
        laser_points (torch.Tensor): Laser points on the visible wall [B, 3, L]
        volume_def (VolumeDef): Definition of the volume
        bin_def (BinDef): Bin width and offset
        scan_origin (torch.Tensor, optional): Position of the detector for each scan point [B, 3, S]
            Shape may also be [B, 3, 1] or [B, 3]
            If None, the time of flight from the detector to the wall will be ignored.
        laser_origin (torch.Tensor, optional): Position of the laser for each laser point [B, 3, S]
            If None, the time of flight from the laser to the wall will be ignored.

    Returns:
        torch.Tensor: Volume [B, D, H, W]
    """

    @staticmethod
    def forward(transient: torch.Tensor, scan_points: torch.Tensor, laser_points: torch.Tensor,
                volume_def: VolumeDef, bin_def: BinDef,
                scan_origin: typing.Optional[torch.Tensor]=None,
                laser_origin: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError('BackprojectionExhaustiveArellano needs to run on CUDA device, '
                               'but no devices are available.')
        batch_size = transient.shape[0]
        volume = torch.zeros((batch_size,) + volume_def.resolution, device='cuda')
        if scan_origin is None:
            scan_origin = torch.empty_like(scan_points[:,:,:0])
        elif scan_origin.ndim == 2:
            scan_origin = scan_origin[:,:,None].expand_as(scan_points)
        elif scan_origin.ndim == 3:
            scan_origin = scan_origin.expand_as(scan_points)
        else:
            raise ValueError(f'Shape {scan_origin.shape} of scan_origin is invalid.')
        if laser_origin is None:
            laser_origin = torch.empty_like(laser_points[:,:,:0])
        elif laser_origin.ndim == 2:
            laser_origin = laser_origin[:,:,None].expand_as(laser_points)
        elif laser_origin.ndim == 3:
            laser_origin = laser_origin.expand_as(laser_points)
        else:
            raise ValueError(f'Shape {laser_origin.shape} of laser_origin is invalid.')
        backprojection_exhaustive_arellano_forward(
            transient.cuda(), scan_points.cuda(), laser_points.cuda(),
            scan_origin.cuda(), laser_origin.cuda(),
            volume,
            volume_def.start, volume_def.end,
            bin_def.width, bin_def.offset
        )
        return volume.to(transient.device)

class BackprojectionConfocalVelten(torch.autograd.Function):
    """Compute probability volume by backprojecting transient points.

    Implementation follows the one proposed in:
    Velten, A., Willwacher, T., Gupta, O., Veeraraghavan, A., Bawendi, M. G., & Raskar, R. (2012).
    Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging.
    Nature communications, 3(1), 1-8.

    The scan and laser points and positions as well as the volume start and end points (x,y,z)
    correspond to (W,H,D). The volume resolution on the other hand is given as [D, H, W].

    Args:
        transient (torch.Tensor): Transient measurement volume [B, T, S]
        scan_points (torch.Tensor): Scan/laser points on the visible wall [B, S, 3]
        volume_def (VolumeDef): Definition of the volume
        bin_def (BinDef): Bin width and offset
        scan_origin (torch.Tensor, optional): Position of the detector and laser for each scan point
            [B, S, 3] Shape may also be [B, 1, 3] or [B, 3]
            If None, the time of flight from the detector to the wall will be ignored.

    Returns:
        torch.Tensor: Volume [B, D, H, W]
    """

    @staticmethod
    def forward(ctx, transient: torch.Tensor, scan_points: torch.Tensor,
                volume_def: VolumeDef, bin_def: BinDef,
                scan_origin: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError('BackprojectionConfocalVelten needs to run on CUDA device, '
                               'but no devices are available.')
        batch_size = transient.shape[0]
        volume = torch.zeros((batch_size,) + volume_def.resolution, device='cuda')
        scan_origin = origin_of_wallpoints(scan_origin, scan_points, 'scan_origin')
        backprojection_confocal_velten_forward(
            transient.cuda(), scan_points.cuda(), scan_origin.cuda(), volume,
            volume_def.start, volume_def.end,
            bin_def.width, bin_def.offset
        )
        return volume.to(transient.device)

class BackprojectionExhaustiveVelten(torch.autograd.Function):
    """Compute probability volume by backprojecting transient points.

    Implementation follows the one proposed in:
    Velten, A., Willwacher, T., Gupta, O., Veeraraghavan, A., Bawendi, M. G., & Raskar, R. (2012).
    Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging.
    Nature communications, 3(1), 1-8.

    The scan and laser points and positions as well as the volume start and end points (x,y,z)
    correspond to (W,H,D). The wall is assumed to be at the xy plane for some
    fixed z. The volume resolution on the other hand is given as [D, H, W].

    Args:
        transient (torch.Tensor): Transient measurement volume [B, N, S, L]
        scan_points (torch.Tensor): Scan points on the visible wall [B, S, 3]
        laser_points (torch.Tensor): Laser points on the visible wall [B, L, 3]
        volume_def (VolumeDef): Definition of the volume
        bin_def (BinDef): Bin width and offset
        scan_origin (torch.Tensor, optional): Position of the detector for each scan point [B, S, 3]
            Shape may also be [B, 1, 3] or [B, 3]. If None, the time of flight from the detector to
            the wall will be ignored.
        laser_origin (torch.Tensor, optional): Position of the laser for each laser point [B, L, 3]
            Shape may also be [B, 1, 3] or [B, 3]. If None, the time of flight from the laser to
            the wall will be ignored.

    Returns:
        torch.Tensor: Volume [B, D, H, W]
    """

    @staticmethod
    def forward(ctx, transient: torch.Tensor, scan_points: torch.Tensor, laser_points: torch.Tensor,
                volume_def: VolumeDef, bin_def: BinDef,
                scan_origin: typing.Optional[torch.Tensor]=None,
                laser_origin: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError('BackprojectionExhaustiveVelten needs to run on CUDA device, '
                               'but no devices are available.')
        batch_size = transient.shape[0]
        volume = torch.zeros((batch_size,) + volume_def.resolution, device='cuda')
        scan_origin = origin_of_wallpoints(scan_origin, scan_points, 'scan_origin')
        laser_origin = origin_of_wallpoints(laser_origin, laser_points, 'laser_origin')
        backprojection_exhaustive_velten_forward(
            transient.cuda(), scan_points.cuda(), laser_points.cuda(),
            scan_origin.cuda(), laser_origin.cuda(),
            volume,
            volume_def.start, volume_def.end,
            bin_def.width, bin_def.offset
        )
        return volume.to(transient.device)
