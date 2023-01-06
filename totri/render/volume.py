import typing
import torch
from totri.types import VolumeDef, BinDef
from totri_cpp.render import (
    volume_confocal_forward,
    volume_confocal_grad_volume,
    volume_confocal_grad_measurement_points,
    volume_render_exhaustive_forward,
    volume_render_exhaustive_grad_volume,
)

class VolumeRenderConfocal(torch.autograd.Function):

    @staticmethod
    def apply_transposed(transient: torch.Tensor, measurement_points: torch.Tensor,
                volume_bbox_origin: typing.Tuple, volume_bbox_size: typing.Tuple,
                volume_resolution: typing.Tuple, bin_width: float, bin_offset: float):
        volume = torch.empty(volume_resolution, device='cuda')
        volume_confocal_grad_volume(
            volume, measurement_points, transient,
            volume_bbox_origin, volume_bbox_size,
            bin_width, bin_offset)
        return volume

    @staticmethod
    def forward(ctx, volume: torch.Tensor, measurement_points: torch.Tensor,
                volume_bbox_origin: typing.Tuple, volume_bbox_size: typing.Tuple,
                bin_num: int, bin_width: float, bin_offset: float):
        if volume.ndim != 3:
            raise ValueError('Volume tensor must be 3d')
        if not volume.is_cuda:
            raise NotImplementedError('VolumeRenderConfocal is only implemented for cuda tensors')
        if not measurement_points.is_cuda:
            raise NotImplementedError('VolumeRenderConfocal is only implemented for cuda tensors')
        if measurement_points.ndim != 2:
            raise ValueError('Measurement Point Tensor must be 2d')
        if measurement_points.shape[0] != 3:
            raise ValueError('Measurement Point Tensor must be 3xn')
        if len(volume_bbox_origin) != 3:
            raise ValueError('volume_bbox_origin must have 3 entries')
        if len(volume_bbox_size) != 3:
            raise ValueError('volume_bbox_size must have 3 entries')
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(volume, measurement_points)
        ctx.volume_bbox_origin = volume_bbox_origin
        ctx.volume_bbox_size = volume_bbox_size
        ctx.bin_width = bin_width
        ctx.bin_offset = bin_offset
        transient = torch.zeros((measurement_points.shape[1], bin_num), device='cuda', dtype=volume.dtype)
        volume_confocal_forward(
            volume, measurement_points, transient,
            volume_bbox_origin, volume_bbox_size,
            bin_width, bin_offset)
        return transient

    @staticmethod
    def backward(ctx, transient_grad: torch.Tensor):
        if transient_grad is None:
            return (None,)*7
        volume_grad = None
        measurement_points_grad = None
        if ctx.needs_input_grad[0]:
            volume, measurement_points = ctx.saved_tensors
            volume_grad = torch.empty_like(volume)
            volume_confocal_grad_volume(
                volume_grad, measurement_points, transient_grad,
                ctx.volume_bbox_origin, ctx.volume_bbox_size,
                ctx.bin_width, ctx.bin_offset)
        if ctx.needs_input_grad[1]:
            volume, measurement_points = ctx.saved_tensors
            measurement_points_grad = torch.empty_like(measurement_points)
            volume_confocal_grad_measurement_points(
                measurement_points_grad, volume, measurement_points, transient_grad,
                ctx.volume_bbox_origin, ctx.volume_bbox_size,
                ctx.bin_width, ctx.bin_offset)
        return (volume_grad, measurement_points_grad) + (None,)*5

class VolumeRenderExhaustive(torch.autograd.Function):
    """Render occupancy volume to transient image.

    The scan and laser points and positions as well as the volume start and end points (x,y,z)
    correspond to (W,H,D). The wall is assumed to be at the xy plane for some
    fixed z. The volume resolution on the other hand is given as [D, H, W].

    Args:
        volume (torch.Tensor): Occupancy volume [B, D, H, W]
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
        torch.Tensor: Transient image [B, N, S, L]
    """

    @staticmethod
    def forward(ctx, volume: torch.Tensor, scan_points: torch.Tensor, laser_points: torch.Tensor,
                volume_def: VolumeDef, bin_def: BinDef,
                scan_origin: typing.Optional[torch.Tensor]=None,
                laser_origin: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError('VolumeRenderExhaustive needs to run on CUDA device, '
                               'but no devices are available.')
        batch_size = volume.shape[0]
        transient = torch.zeros(
            (batch_size, bin_def.num, scan_points.shape[2], laser_points.shape[2]),
            device='cuda')
        if scan_origin is None:
            scan_origin = torch.empty_like(scan_points[:,:,:0])
        elif scan_origin.ndim == 2:
            scan_origin = scan_origin[:,:,None].expand_as(scan_points)
        elif scan_origin.ndim == 3:
            if scan_origin.shape[2] > 0:
                scan_origin = scan_origin.expand_as(scan_points)
        else:
            raise ValueError(f'Shape {scan_origin.shape} of scan_origin is invalid.')
        if laser_origin is None:
            laser_origin = torch.empty_like(laser_points[:,:,:0])
        elif laser_origin.ndim == 2:
            laser_origin = laser_origin[:,:,None].expand_as(laser_points)
        elif laser_origin.ndim == 3:
            if laser_origin.shape[2] > 0:
                laser_origin = laser_origin.expand_as(laser_points)
        else:
            raise ValueError(f'Shape {laser_origin.shape} of laser_origin is invalid.')
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(volume, scan_points, laser_points, scan_origin, laser_origin)
        ctx.volume_def = volume_def
        ctx.bin_def = bin_def
        volume_render_exhaustive_forward(
            volume.cuda(), scan_points.cuda(), laser_points.cuda(),
            scan_origin.cuda(), laser_origin.cuda(),
            transient,
            volume_def.start, volume_def.end,
            bin_def.width, bin_def.offset)
        return transient.to(volume.device)

    @staticmethod
    def backward(ctx, transient_grad: torch.Tensor):
        grad = [None, None, None, None, None, None, None]
        if transient_grad is None:
            return grad
        if ctx.needs_input_grad[0]: # volume
            volume, scan_points, laser_points, scan_origin, laser_origin = ctx.saved_tensors
            volume_grad = torch.empty_like(volume, device='cuda')
            volume_render_exhaustive_grad_volume(
                transient_grad.cuda(), scan_points.cuda(), laser_points.cuda(),
                scan_origin.cuda(), laser_origin.cuda(),
                volume_grad,
                ctx.volume_def.start, ctx.volume_def.end,
                ctx.bin_def.width, ctx.bin_def.offset)
            grad[0] = volume_grad.to(transient_grad.device())
        if ctx.needs_input_grad[1]: # scan_points
            raise NotImplementedError()
        if ctx.needs_input_grad[2]: # laser_points
            raise NotImplementedError()
        if ctx.needs_input_grad[3]: # volume_def
            raise NotImplementedError()
        if ctx.needs_input_grad[4]: # bin_Def
            raise NotImplementedError()
        if ctx.needs_input_grad[5]: # scan_origin
            raise NotImplementedError()
        if ctx.needs_input_grad[6]: # laser_origin
            raise NotImplementedError()
        return grad
