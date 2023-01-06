import typing
import torch
from totri.types import BinDef
from totri.util import faces_of_verts, colors_of_verts, origin_of_wallpoints
from totri_cpp.render import (
    mesh_confocal_forward,
    mesh_confocal_grad_verts,
    mesh_exhaustive_forward,
    mesh_exhaustive_grad_verts,
)

class MeshRenderConfocal(torch.autograd.Function):
    """Render mesh to transient image.

    Args:
        verts (torch.Tensor): Vertices [B, V, 3]
        faces (torch.Tensor, optional): Face indices [B, F, 3]
        scan_points (torch.Tensor): Laser/Scan points on the wall [B, S, 3]
        bin_def (BinDef): Bin count, width and offset
        colors (torch.Tensor, optional): Color of each vertex [B, V, 1].
            If None, the color will be set as 1.
        scan_origin (torch.Tensor, optional): Position of the laser/detector for
            each scan point [B, S, 3]. Shape may also be [B, 1, 3] or [B, 3].
            If None, the time of flight from the detector to the wall will be
            ignored.

    Returns:
        torch.Tensor: Transient image [B, T, S]
    """

    @staticmethod
    def forward(ctx, verts: torch.Tensor, faces:  typing.Optional[torch.Tensor],
                scan_points: torch.Tensor, bin_def: BinDef,
                colors: typing.Optional[torch.Tensor] = None,
                scan_origin: typing.Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        faces = faces_of_verts(verts, faces)
        colors = colors_of_verts(colors, verts)
        scan_origin = origin_of_wallpoints(scan_origin, scan_points,
                                           'scan_origin')
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(verts, faces, colors, scan_points, scan_origin)
        ctx.bin_def = bin_def
        transient = torch.zeros(
            (1, bin_def.num, scan_points.shape[1]),
            device='cuda')
        mesh_confocal_forward(verts, faces, colors, scan_points, scan_origin,
                              transient, bin_def.width, bin_def.offset, 0)
        return transient

    @staticmethod
    def backward(ctx, transient_grad: torch.Tensor):
        grad = [None, None, None, None, None, None]
        if transient_grad is None:
            return tuple(grad)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[4]: # verts or colors
            verts, faces, colors, scan_points, scan_origin = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                verts_grad = torch.empty_like(verts)
            else:
                verts_grad = torch.empty_like(verts[:,:,:0])
            if ctx.needs_input_grad[4]:
                colors_grad = torch.empty_like(colors)
            else:
                colors_grad = torch.empty_like(colors[:,:,:0])
            mesh_confocal_grad_verts(verts, faces, colors,
                                     scan_points, scan_origin,
                                     transient_grad, verts_grad, colors_grad,
                                     ctx.bin_def.width, ctx.bin_def.offset, 0)
            if ctx.needs_input_grad[0]:
                grad[0] = verts_grad
            if ctx.needs_input_grad[4]:
                grad[4] = colors_grad
        if ctx.needs_input_grad[1]: # faces
            raise NotImplementedError()
        if ctx.needs_input_grad[2]: # scan_points
            raise NotImplementedError()
        if ctx.needs_input_grad[3]: # bin_def
            raise NotImplementedError()
        if ctx.needs_input_grad[5]: # scan_origin
            raise NotImplementedError()
        return tuple(grad)

class MeshRenderExhaustive(torch.autograd.Function):
    """Render mesh to transient image.

    Args:
        verts (torch.Tensor): Vertices [B, V, 3]
        faces (torch.Tensor, optional): Face indices [B, F, 3]
        scan_points (torch.Tensor): Scan points on the visible wall [B, S, 3]
        laser_points (torch.Tensor): Laser points on the visible wall [B, L, 3]
        bin_def (BinDef): Bin count, width and offset
        colors (torch.Tensor, optional): Color of each vertex [B, V, 1].
            If None, the color will be set as 1.
        scan_origin (torch.Tensor, optional): Position of the Detector for each 
            scan point [B, S, 3]. Shape may also be [B, 1, 3] or [B, 3].
            If None, the time of flight from the detector to the wall will be
            ignored.
        laser_origin (torch.Tensor, optional): Position of the Laser for each 
            laser point [B, S, 3]. Shape may also be [B, 1, 3] or [B, 3].
            If None, the time of flight from the laser to the wall will be
            ignored.

    Returns:
        torch.Tensor: Transient image [B, T, S, L]
    """

    @staticmethod
    def forward(ctx, verts: torch.Tensor, faces: typing.Optional[torch.Tensor],
                scan_points: torch.Tensor, laser_points: torch.Tensor,
                bin_def: BinDef, colors: typing.Optional[torch.Tensor] = None,
                scan_origin: typing.Optional[torch.Tensor] = None,
                laser_origin: typing.Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        faces = faces_of_verts(verts, faces)
        colors = colors_of_verts(colors, verts)
        scan_origin = origin_of_wallpoints(scan_origin, scan_points,
                                           'scan_origin')
        laser_origin = origin_of_wallpoints(laser_origin, laser_points,
                                            'laser_points')
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(verts, faces, colors, scan_points, laser_points,
                              scan_origin, laser_origin)
        ctx.bin_def = bin_def
        transient = torch.zeros(
            (1, bin_def.num, scan_points.shape[1], laser_points.shape[1]),
            device='cuda')
        mesh_exhaustive_forward(
            verts, faces, colors, scan_points, laser_points, scan_origin,
            laser_origin, transient, bin_def.width, bin_def.offset, 0)
        return transient

    @staticmethod
    def backward(ctx, transient_grad: torch.Tensor):
        grad = [None, None, None, None, None, None, None, None]
        if transient_grad is None:
            return tuple(grad)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[5]: # verts or colors
            verts, faces, colors, scan_points, laser_points, scan_origin, laser_origin = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                verts_grad = torch.empty_like(verts)
            else:
                verts_grad = torch.empty_like(verts[:,:,:0])
            if ctx.needs_input_grad[5]:
                colors_grad = torch.empty_like(colors)
            else:
                colors_grad = torch.empty_like(colors[:,:,:0])
            mesh_exhaustive_grad_verts(
                verts, faces, colors, scan_points, laser_points, scan_origin,
                laser_origin, transient_grad, verts_grad, colors_grad,
                ctx.bin_def.width, ctx.bin_def.offset, 0)
            if ctx.needs_input_grad[0]:
                grad[0] = verts_grad
            if ctx.needs_input_grad[5]:
                grad[5] = colors_grad
        if ctx.needs_input_grad[1]: # faces
            raise NotImplementedError()
        if ctx.needs_input_grad[2]: # scan_points
            raise NotImplementedError()
        if ctx.needs_input_grad[3]: # laser_points
            raise NotImplementedError()
        if ctx.needs_input_grad[4]: # bin_def
            raise NotImplementedError()
        if ctx.needs_input_grad[6]: # scan_origin
            raise NotImplementedError()
        if ctx.needs_input_grad[7]: # laser_origin
            raise NotImplementedError()
        return tuple(grad)
