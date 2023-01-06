import typing
import torch
from totri.types import VolumeDef
from totri_cpp.util import (
    rbf2volume_gaussian_forward,
    rbf2volume_gaussian_grad
)

class Rbf2VolumeGaussian(torch.autograd.Function):
    """Gaussian Rbf to Volume

    Args:
        rbf (torch.Tensor): Parameters (position, sigma, attributes) [B, N, 4+A]
        volume_def (VolumeDef): Shape of the volume

    Returns:
        torch.Tensor: Volume [B, 1+A, D, H, W]
    """

    @staticmethod
    def forward(ctx, rbf: torch.Tensor, volume_def: VolumeDef) -> torch.Tensor:
        num_attributes = rbf.shape[2] - 4
        volume_shape = (rbf.shape[0], 1 + num_attributes) + volume_def.resolution
        if rbf.shape[1] == 0:
            volume = torch.zeros(volume_shape, device='cuda', dtype=rbf.dtype)
        else:
            volume = torch.empty(volume_shape, device='cuda', dtype=rbf.dtype)
            rbf2volume_gaussian_forward(
                rbf, volume,
                volume_def.start, volume_def.end)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(rbf, volume)
        ctx.volume_def = volume_def
        return volume

    @staticmethod
    def backward(ctx, volume_grad: torch.Tensor):
        rbf_grad = None
        if volume_grad is None:
            return (rbf_grad, None)
        if ctx.needs_input_grad[0]:
            rbf, volume = ctx.saved_tensors
            rbf_grad = torch.zeros_like(rbf)
            if rbf.shape[1] > 0:
                rbf2volume_gaussian_grad(
                    rbf, volume, volume_grad, rbf_grad,
                    ctx.volume_def.start, ctx.volume_def.end)
        if ctx.needs_input_grad[1]: # volume_def
            raise NotImplementedError()
        return (rbf_grad, None)
