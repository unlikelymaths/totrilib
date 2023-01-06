"""DepthMapFitting"""
from configparser import Interpolation
import typing
import torch
import torchvision
from totri.types import BinDef, VolumeDef
from totri.render import MeshRenderConfocal, MeshRenderExhaustive
from totri.fitting.background import (
    TransientBackgroundConfocal,
    TransientBackgroundExhaustive)
from totri.util import tensor_to_chw, TotriOptimizer, MultiplicativeInterpolator, LinearInterpolator, Interpolator

def verts_transform(verts):
    return torch.stack((
        -verts[:,:,0],
        verts[:,:,1],
        -verts[:,:,2]),
        dim=-1)

def img_transform(img):
    return img.flip([0, 1])

def dx_dy(tensor: torch.Tensor) -> typing.Tuple[torch.Tensor]:
    dx = tensor[:,1:] - tensor[:,:-1]
    dy = tensor[1:,:] - tensor[:-1,:]
    dx = torch.cat((dx, torch.zeros_like(dx[:,:1])), dim=1)
    dy = torch.cat((dy, torch.zeros_like(dy[:1,:])), dim=0)
    return (dx, dy)

def tv_loss(tensor):
    eps = 1.e-3
    dx, dy = dx_dy(tensor)
    return torch.mean((dx*dx + dy*dy + eps)**0.5 - eps**0.5)

def w_loss(tensor):
    # return ((1 - (2*((tensor+1.e-6)**0.5) - 1)**2).clamp(min=0)).mean()
    split = 0.5
    return torch.where(
        tensor < split,
        tensor / split,
        1 - (tensor - split) / (1 - split)
        ).clamp(min=0).mean()

def lerp(iteration, num_iter, values):
    if isinstance(values, tuple) or isinstance(values, list):
        if len(values) == 1:
            return values[0]
        elif len(values) == 2:
            weight = max(min(iteration / (num_iter-1), 1), 0)
            return (1 - weight) * values[0] + weight * values[1]
        else:
            raise NotImplementedError()
    return values

def get_lr_fun(warmup, num_iter, values):
    def lr_fun(iteration):
        if iteration < warmup:
            weight = ((iteration + 1) / warmup)
        else:
            weight = 1
        return weight * lerp(iteration, num_iter, values)
    return lr_fun

class DepthMap():

    def __init__(self, volume_def: VolumeDef):
        self.volume_def = volume_def
        self.depth_scaling = 0.5 * abs(self.volume_def.end[2] - self.volume_def.start[2])
        """VolumeDef"""
        self.depth = torch.full(
            (volume_def.resolution[1], volume_def.resolution[2]),
            0.5,
            device='cuda'
        )
        """depth [H, W]"""
        self.offset = torch.full(
            (1,),
            0.5 * (self.volume_def.start[2] + self.volume_def.end[2]),
            device='cuda')
        """offset [1,]"""
        self.color = torch.full(
            (volume_def.resolution[1], volume_def.resolution[2]),
            1.0,
            device='cuda'
        )
        """color [H, W]"""
        self.xy_coordinates = self.get_xy_coords()
        """xy coordinates [H, W, 2]"""
        self.faces = self.get_faces()
        """face indices [F, 3]"""
        self.clamp_params(None)

    def requires_grad(self, requires_grad):
        self.depth.requires_grad = requires_grad
        self.offset.requires_grad = requires_grad
        self.color.requires_grad = requires_grad

    def params(self):
        return [
            tensor
            for tensor in [self.depth, self.offset, self.color]
            if tensor.requires_grad
        ]

    def clamp_params(self, intensity):
        with torch.no_grad():
            color_scaling = 1.0 / (1.e-12 + self.color.max())
            if intensity is not None:
                intensity[0] = intensity[0] / color_scaling
            self.color[:]     = torch.clamp(self.color[:] * color_scaling, min=1.e-12)
            self.color[:,  0] = 1.e-12
            self.color[:, -1] = 1.e-12
            self.color[ 0, :] = 1.e-12
            self.color[-1, :] = 1.e-12
            self.depth[:]     = torch.clamp(self.depth[:], min=0, max=1)
            self.depth[:,  0] = 0.5
            self.depth[:, -1] = 0.5
            self.depth[ 0, :] = 0.5
            self.depth[-1, :] = 0.5

    def abs_depth(self):
        """Transform depth to absolute value

        Returns:
            torch.Tensor: depth [H, W]
        """
        return self.offset - self.depth_scaling * (self.depth - 0.5)

    def verts(self) -> torch.Tensor:
        """Build vertex tensor

        Returns:
            torch.Tensor: vertices [N, 3]
        """
        return torch.cat((self.xy_coordinates, self.abs_depth()[:,:,None]), dim=2).view(-1, 3)

    def colors(self) -> torch.Tensor:
        """Build color tensor

        Returns:
            torch.Tensor: vertices [N, 3]
        """
        return self.color[:,:,None].expand(-1, -1, 3).view(-1, 3)

    def resize(self, resolution):
        self.volume_def.resolution = (
            self.volume_def.resolution[0],
            resolution[0],
            resolution[1])
        self.depth = torch.nn.functional.interpolate(
            self.depth[None, None, :, :],
            size=resolution,
            mode="bilinear",
            align_corners=False)[0,0]
        self.color = torch.nn.functional.interpolate(
            self.color[None, None, :, :],
            size=resolution,
            mode="bilinear",
            align_corners=False)[0,0]
        self.xy_coordinates = self.get_xy_coords()
        self.faces = self.get_faces()

    def blur(self, kernel_size):
        self.depth[:] = torchvision.transforms.functional.gaussian_blur(
            self.depth[None], kernel_size)[0]
        self.color[:] = torchvision.transforms.functional.gaussian_blur(
            self.color[None], kernel_size)[0]

    def get_xy_coords(self):
        x = torch.linspace(
            self.volume_def.start[0], self.volume_def.end[0],
            self.volume_def.resolution[2], device='cuda')
        y = torch.linspace(
            self.volume_def.start[1], self.volume_def.end[1],
            self.volume_def.resolution[1], device='cuda')
        return torch.stack(torch.meshgrid(y, x)[::-1], dim=-1)

    def get_faces(self):
        faces = []
        width = self.volume_def.resolution[2]
        height = self.volume_def.resolution[1]
        def idx(x, y):
            return x + y * width
        for y in range(height-1):
            for x in range(width-1):
                faces.append([
                    idx(x  , y  ),
                    idx(x+1, y+1),
                    idx(x+1, y  )])
                faces.append([
                    idx(x  , y  ),
                    idx(x  , y+1),
                    idx(x+1, y+1)])
        return torch.tensor(faces, dtype=torch.int32, device='cuda')

class DepthMapFitting():

    def __init__(self, bin_def: BinDef,
                 tensorboard: bool, log_dir: str = None,
                 log_val_interval: int = 1,
                 log_img_interval: int = 50):
        self.bin_def = bin_def
        self.tensorboard = tensorboard
        self.log_dir = log_dir
        self.summary_writer = None
        self.log_val_interval = log_val_interval
        self.log_img_interval = log_img_interval
        self.iter = 0
        self.intensity = None

    def fit(self, depth_map: DepthMap, transient_input: torch.Tensor,
            scan_points: torch.Tensor,
            laser_points: typing.Optional[torch.nn.Module] = None,
            scan_origin: typing.Optional[torch.nn.Module] = None,
            laser_origin: typing.Optional[torch.nn.Module] = None,
            background_model: typing.Optional[torch.nn.Module] = None,
            scan_point_limit = 0,
            num_iter: int = 2000,
            lr_interpolator: Interpolator = MultiplicativeInterpolator(1.e-2, 1.e-4, gamma=0.99),
            lambda_depth = 0.2,
            lambda_color = 0.1,
            lambda_w = 0.05,
            lambda_black = 0,
            warmup = 0,
            subdivide_at = [],
            callback = None,
            ):
        """Fit depth map to data

        Args:
            transient_input (torch.Tensor): [T, S] or [T, S, L]
            scan_points (torch.Tensor): [S, 3]

        Returns:
            DepthMap: Fitted depth map
        """
        scan_points = scan_points[None]
        if laser_points is not None:
            laser_points = laser_points[None]
        if scan_origin is not None:
            scan_origin = scan_origin[None]
        if laser_origin is not None:
            laser_origin = laser_origin[None]
        if self.tensorboard and self.summary_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(
                log_dir=self.log_dir
            )

        transient_input = transient_input / transient_input.mean()
        max_val = torch.max(transient_input)
        if scan_point_limit > 0:
            num_scan_points = transient_input.shape[1]
            subset = torch.randperm(num_scan_points, device='cuda')
            transient_input_full = transient_input
            scan_points_full = scan_points
            transient_input_subsets = []
            scan_point_subsets = []
            for start in range(0, num_scan_points, scan_point_limit):
                end = start + scan_point_limit
                transient_input_subsets.append(
                    transient_input_full[:,subset[start:end]].contiguous())
                scan_point_subsets.append(
                    scan_points[:,subset[start:end]].contiguous())
            transient_input = transient_input_subsets[0]
            scan_points = scan_point_subsets[0]

        if self.summary_writer is not None:
            if transient_input.ndim == 2:
                self.summary_writer.add_image(
                    'img/reference', 
                    transient_input / max_val,
                    dataformats='HW')
            else:
                self.summary_writer.add_image(
                    'img/reference',
                    tensor_to_chw(transient_input / max_val, 'HWN'),
                    dataformats='CHW')

        # Intensity
        if self.intensity is None:
            if laser_points is None:
                transient_rendered = MeshRenderConfocal.apply(
                        depth_map.verts()[None], depth_map.faces[None], scan_points,
                        self.bin_def, depth_map.color.view(1, -1, 1), scan_origin)[0]
            else:
                transient_rendered = MeshRenderExhaustive.apply(
                        depth_map.verts()[None], depth_map.faces[None], scan_points, laser_points,
                        self.bin_def, depth_map.color.view(1, -1, 1), scan_origin, laser_origin)[0]
            num = (transient_rendered * transient_input).sum()
            den = (transient_rendered * transient_rendered).sum()
            self.intensity = (num / den.clamp(min=1.e-6)).clamp(min=1.e-6).view(1,)
            self.intensity.requires_grad = True

        depth_map.requires_grad(True)
        if background_model is None:
            background_parameters = list()
        else:
            background_model.requires_grad_ = True
            background_parameters = list(background_model.parameters())
        optim = TotriOptimizer(
            depth_map.params() + [self.intensity,] + background_parameters,
            lr_interpolator,
            warmup)

        for i in range(num_iter):
            if i%100 == 0:
                print(f'{i}/{num_iter}')
            if self.iter in subdivide_at:
                res = depth_map.volume_def.resolution[1:]
                depth_map.requires_grad(False)
                depth_map.resize((res[0]*2, res[1]*2))
                depth_map.blur(4*2+1)
                depth_map.requires_grad(True)
                optim.set_params(
                    depth_map.params() + [self.intensity,] + background_parameters,
                    warmup)
                # lambda_depth *= 2
                # lambda_color *= 2
            # elif self.iter % 10 == 5:
            #     with torch.no_grad():
            #         depth_map.blur(5)
            optim.zero_grad()

            if scan_point_limit > 0:
                transient_input = transient_input_subsets[i % len(transient_input_subsets)]
                scan_points = scan_point_subsets[i % len(transient_input_subsets)]

            # Render transient
            if laser_points is None:
                transient_rendered = MeshRenderConfocal.apply(
                        depth_map.verts()[None], depth_map.faces[None], scan_points,
                        self.bin_def, depth_map.color.view(1, -1, 1), scan_origin)[0]
            else:
                transient_rendered = MeshRenderExhaustive.apply(
                        depth_map.verts()[None], depth_map.faces[None], scan_points, laser_points,
                        self.bin_def, depth_map.color.view(1, -1, 1), scan_origin, laser_origin)[0]
            # Compute background
            if background_model is None:
                transient = transient_rendered
            else:
                if laser_points is None:
                    transient_background = background_model(scan_points)[0]
                else:
                    transient_background = background_model(scan_points, laser_points)[0]
                transient = transient_rendered + transient_background
            # Normalize
            transient = self.intensity * transient
            # Losses
            data_loss = 10.0 * torch.mean((transient - transient_input)**2)
            depth_reg_loss = lambda_depth * 1000.0 * tv_loss(depth_map.depth)
            color_reg_loss = lambda_color * 10.0 * tv_loss(depth_map.color)
            w_reg_loss = 0 # lerp(i, num_iter, lambda_w) * 10.0 * w_loss(depth_map.color)
            black_reg_loss =  lambda_black * 10.0 * depth_map.color.mean()
            loss = data_loss + depth_reg_loss + color_reg_loss + w_reg_loss + black_reg_loss
            loss.backward()
            optim.step()
            self.iter += 1

            depth_map.clamp_params(self.intensity)
            if background_model is not None:
                background_model.clamp_parameters(transient_rendered, transient_background)

            if callback is not None:
                callback(self.iter, depth_map, transient, loss)

            if self.summary_writer is not None and self.iter % self.log_val_interval == 0:
                self.summary_writer.add_scalar('loss/loss', loss, self.iter)
                self.summary_writer.add_scalar('loss/data_loss', data_loss, self.iter)
                self.summary_writer.add_scalar('loss/depth_reg_loss', depth_reg_loss, self.iter)
                self.summary_writer.add_scalar('loss/color_reg_loss', color_reg_loss, self.iter)
                self.summary_writer.add_scalar('loss/w_reg_loss', w_reg_loss, self.iter)
                self.summary_writer.add_scalar('loss/black_reg_loss', black_reg_loss, self.iter)
                self.summary_writer.add_scalar('param/lr', optim.lr, self.iter)
                self.summary_writer.add_scalar('param/lambda_depth', lerp(i, num_iter, lambda_depth), self.iter)
                self.summary_writer.add_scalar('param/lambda_color', lerp(i, num_iter, lambda_color), self.iter)
                self.summary_writer.add_scalar('param/lambda_w', lerp(i, num_iter, lambda_w), self.iter)
                self.summary_writer.add_scalar('param/lambda_black', lerp(i, num_iter, lambda_black), self.iter)
            if self.summary_writer is not None and self.iter % self.log_img_interval == 0:
                if transient_input.ndim == 2:
                    self.summary_writer.add_image(
                        'img/transient', 
                        (transient / max_val).clamp(max=1),
                        self.iter, dataformats='HW')
                    self.summary_writer.add_image(
                        'img/rendered', 
                        (transient_rendered / transient_rendered.max()).clamp(max=1),
                        self.iter, dataformats='HW')
                    if background_model is not None:
                        self.summary_writer.add_image(
                            'img/background', 
                            (transient_background / transient_background.max()).clamp(max=1),
                            self.iter, dataformats='HW')
                    self.summary_writer.add_image(
                        'img/error', 
                        ((transient - transient_input).abs() / max_val).clamp(max=1),
                        self.iter, dataformats='HW')
                else:
                    self.summary_writer.add_image(
                        'img/transient',
                        tensor_to_chw((transient / max_val).clamp(max=1), 'HWN'),
                        self.iter, dataformats='CHW')
                    self.summary_writer.add_image(
                        'img/rendered',
                        tensor_to_chw((transient_rendered / transient_rendered.max()).clamp(max=1), 'HWN'),
                        self.iter, dataformats='CHW')
                    if background_model is not None:
                        self.summary_writer.add_image(
                            'img/background',
                            tensor_to_chw((transient_background / transient_background.max()).clamp(max=1), 'HWN'),
                            self.iter, dataformats='CHW')
                    self.summary_writer.add_image(
                        'img/error',
                        tensor_to_chw(((transient - transient_input).abs() / max_val).clamp(max=1), 'HWN'),
                        self.iter, dataformats='CHW')
                self.summary_writer.add_image(
                    'img/depth', 
                    img_transform(depth_map.depth),
                    self.iter, dataformats='HW')
                self.summary_writer.add_image(
                    'img/color', 
                    img_transform(depth_map.color),
                    self.iter, dataformats='HW')
                self.summary_writer.add_mesh(
                    'mesh/mesh',
                    verts_transform(depth_map.verts()[None]),
                    faces=depth_map.faces[None],
                    colors=255*depth_map.colors()[None],
                    global_step=self.iter)
        depth_map.requires_grad(False)
