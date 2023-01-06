import os
import abc
import copy
import typing
from cv2 import threshold
import torch
from tomcubes import MarchingCubesFunction
from totri.types import C0, BinDef, VolumeDef
from totri.render import MeshRenderConfocal, MeshRenderExhaustive
from totri.util import RealTimeVideoWriter
from totri.util.render import UnitCubeMeshRender
from totri.util.format import verts_transform, faces_of_verts
from totri.reconstruct import (
    BackprojectionConfocalVelten,
    BackprojectionExhaustiveVelten,
    second_derivative_filter)
from totri.util import Rbf2VolumeGaussian, tensor_to_chw
from totri.fitting.background import TransientBackgroundBase, TransientBackgroundEmpty

class RbfBase(abc.ABC):

    def __init__(self, volume_def: VolumeDef, num_channels: int, has_color: bool):
        self.volume_def = volume_def
        """Volume containing the rbf base"""
        self.num_channels = num_channels
        """Number of channels to represent each rbf"""
        self.has_color = has_color
        """True if the base keeps track of color"""
        self.params = torch.empty((0, self.num_channels), device='cuda')
        """Rbf parameters [N, 4(+A)] with entries [x, y, z, sigma, (attributes)]"""
        self.volume_mask_ = None
        """Binary volume mask [D, H, W]"""

    def clone(self):
        new_base = copy.copy(self)
        new_base.params = self.params.clone()
        return new_base

    def subdivide(self):
        self.volume_def.resolution = (
            2 * self.volume_def.resolution[0],
            2 * self.volume_def.resolution[1],
            2 * self.volume_def.resolution[2],
        )
        self.volume_mask_ = None

    @abc.abstractmethod
    def difference(self, base):
        pass

    def requires_grad(self, requires_grad: bool):
        self.params.requires_grad = requires_grad

    def param_list(self):
        return [self.params]

    def num_rbf(self):
        return self.params.shape[0]

    def get_pos(self):
        # pos [N, 3]
        return self.params[:,:3]

    def get_sigma(self):
        # pos [N, 1]
        return self.params[:,3:4]

    def get_color(self):
        # color [N, 1] or None
        if self.has_color:
            return self.params[:,3:4]
        else:
            return None

    @abc.abstractmethod
    def clamp_params(self):
        pass

    @abc.abstractmethod
    def make_volume(self) -> torch.Tensor:
        """Make occupancy volume

        Returns:
            torch.Tensor: Volume [1+A, D, H, W]
        """
        pass

    def make_verts(self) -> torch.Tensor:
        """Make vertex tensor

        Returns:
            torch.Tensor: verts [N, 3]
        """
        return MarchingCubesFunction.apply(
            self.make_volume()[None],
            0.5,
            self.volume_def.start, self.volume_def.end)[0]

    @abc.abstractmethod
    def append(self, x: float, y: float, z:float):
        pass

    def remove(self, idx):
        self.params = torch.cat(
            (self.params[:idx], self.params[idx+1:]), dim=0)

    @abc.abstractmethod
    def half_distance(self):
        pass

    def volume_mask(self) -> torch.Tensor:
        """Binary volume mask [D, H, W]"""
        if self.volume_mask_ is None:
            volume_mask = torch.ones(
                self.volume_def.resolution,
                dtype=torch.float32, device='cuda')
            volume_mask[ 0,:,:] = 0
            volume_mask[-1,:,:] = 0
            volume_mask[:, 0,:] = 0
            volume_mask[:,-1,:] = 0
            volume_mask[:,:, 0] = 0
            volume_mask[:,:,-1] = 0 
            self.volume_mask_ = volume_mask
        return self.volume_mask_

class GaussianRbfBase(RbfBase):

    def __init__(self, volume_def: VolumeDef, has_color: bool = False,
                 sigma_init: float = 0.05, sigma_min: float = 0.00001, sigma_max: float = 0.5,
                 margin: float = 0.1):
        num_channels = 4
        if has_color:
            num_channels += 1
        super().__init__(volume_def, num_channels, has_color)
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.margin = margin

    def difference(self, base):
        extent = self.volume_def.extent()
        diff = (self.params - base.params).abs().max(dim=0).values.tolist()
        diff[0] = diff[0] / extent[0]
        diff[1] = diff[1] / extent[1]
        diff[2] = diff[2] / extent[2]
        diff[3] = diff[3] / (self.sigma_max - self.sigma_min)
        # diff[4] already between 1 and 0
        return max(diff)

    def clamp_params(self):
        if self.num_rbf() == 0:
            return
        with torch.no_grad():
            for x in range(3):
                self.params[:,x] = self.params[:,x].clamp(
                    self.volume_def.start[x] + self.margin,
                    self.volume_def.end[x]   - self.margin)
            self.params[:,3] = self.params[:,3].clamp(min=self.sigma_min, max=self.sigma_max)
            if self.has_color:
                if self.num_rbf() > 1:
                    color_scaling = 1.0 / (1.e-12 + self.params[:,4].max())
                    self.params[:,4] = torch.clamp(self.params[:,4] * color_scaling, min=1.e-12)
                else:
                    self.params[:,4] = 1.0

    def make_volume(self) -> torch.Tensor:
        volume = Rbf2VolumeGaussian().apply(self.params[None], self.volume_def)[0]
        volume = torch.cat((volume[:1] * self.volume_mask(), volume[1:]), dim=0)
        return volume

    def append(self, x: float, y: float, z:float):
        if self.has_color:
            new_rbf = torch.tensor(
                [[x, y, z, self.sigma_init, 1.0]],
                dtype=torch.float32, device=self.params.device)
        else:
            new_rbf  = torch.tensor(
                [[x, y, z, self.sigma_init]],
                dtype=torch.float32, device=self.params.device)
        self.params = torch.cat((self.params, new_rbf), dim=0)

    def half_distance(self):
        return 1.177 * self.sigma_init

class RbfFitting():

    def __init__(self, bin_def: BinDef,
                 scan_points: torch.Tensor,
                 laser_points: typing.Optional[torch.Tensor] = None,
                 scan_origin: typing.Optional[torch.Tensor] = None,
                 laser_origin: typing.Optional[torch.Tensor] = None,
                 optimize_check_delete: bool = False,
                 delete_Factor: float = 1.001,
                 tensorboard: bool = False,
                 video: bool = False,
                 log_dir: str = None,
                 project_transient: bool = True):
        self.bin_def_init = bin_def
        self.scan_points = scan_points
        """Scan Points [S, 3]"""
        self.laser_points = laser_points
        """Laser Points [L, 3]"""
        self.scan_origin = scan_origin
        """Scan Origin [S, 3]"""
        self.laser_origin = laser_origin
        """Laser Origin [L, 3]"""
        self.tensorboard = tensorboard
        self.video = video
        if video and not tensorboard:
            raise ValueError('Logging video requires tensorboard output')
        self.log_dir = log_dir
        self.ucmr = UnitCubeMeshRender()
        self.max_samples_per_batch = -1 #4096
        self.optimize_check_delete = optimize_check_delete
        self.delete_Factor = delete_Factor
        self.project_transient = project_transient

    def reduce_transient(self, transient_input, margin=32):
        # [t, s(, l)]
        quantiles = torch.quantile(transient_input.flatten(start_dim=1), 0.95, dim=1).cpu()
        threshold = quantiles.max() / 10
        num_bins = quantiles.shape[0]
        start = 0
        end = num_bins
        for q in range(num_bins-1):
            if quantiles[q] < threshold:
                start = q+1
            else:
                break
        for q in range(num_bins-1, 0, -1):
            if quantiles[q] < threshold:
                end = q
            else:
                break
        start = max(start - margin, 0)
        end = min(end + margin, num_bins)
        self.bin_def = BinDef(
            self.bin_def_init.width,
            self.bin_def_init.offset + self.bin_def_init.width * start,
            end-start)
        transient_input = transient_input[start:end]
        print(f'Removed {start} bins at the beginning and {num_bins-end} bins at the end. {self.bin_def}')
        return transient_input

    def render_visualization(self, base: RbfBase):
        # -> [3, H, W]
        with torch.no_grad():
            verts = base.make_verts()
            if verts.shape[0] == 0:
                return torch.ones((3, 512, 512), device='cuda')
            if base.has_color:
                colors = verts[:,3:4]
            else:
                colors = None
            return self.ucmr.apply(verts[:,:3], colors=colors, volume_def=base.volume_def)

    def render(self, verts, background_model: TransientBackgroundBase, transient_input):
        # verts [t, 3+A] -> [t, s]
        if verts.shape[1] == 4:
            color = torch.clamp(verts[None, :, 3:4].contiguous(), min=1.e-6)
        else:
            color = None
        verts = verts[None, :, :3].contiguous()
        if self.laser_points is None:
            transient_foreground = 10 * MeshRenderConfocal.apply(
                verts, None,
                self.scan_points[None],
                self.bin_def,
                color,
                None if self.scan_origin is None else self.scan_origin[None],
                )[0]
            transient_background = background_model(
                self.scan_points[None],
                )[0]
        else:
            transient_foreground = MeshRenderExhaustive.apply(
                verts, None,
                self.scan_points[None],
                self.laser_points[None],
                self.bin_def,
                color,
                None if self.scan_origin is None else self.scan_origin[None],
                None if self.laser_origin is None else self.laser_origin[None],
                )[0]
            transient_background = background_model(
                self.scan_points[None],
                self.laser_points[None],
                )[0]
        transient = transient_foreground + transient_background
        if self.project_transient:
            num = (transient * transient_input).sum()
            den = (transient * transient).sum()
            # num = (transient * transient_input).sum(dim=0, keepdim=True)
            # den = (transient * transient).sum(dim=0, keepdim=True)
            transient = transient * (num / den.clamp(min=1.e-6)).clamp(min=1.e-6)
        return transient, transient_foreground, transient_background

    def base_to_transient(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input):
        # RbfBase -> [t, s]
        verts = base.make_verts()
        transient, transient_foreground, transient_background = self.render(
            verts, background_model, transient_input)
        return (transient, transient_foreground, transient_background)

    def loss_data(self, transient: torch.Tensor, transient_input: torch.Tensor):
        # [t, s], [t, s] -> [1,]
        return torch.mean((transient - transient_input)**2)

    def loss_blob_size(self, base: RbfBase):
        # RbfBase -> [1,]
        if base.num_rbf() < 1:
            return 0
        sigma = base.get_sigma()
        # return torch.sum((sigma - self.sigma_target)**2)
        return torch.std(sigma)

    def loss_blob_distance(self, base: RbfBase):
        # RbfBase -> [1,]
        if base.num_rbf() <= 1:
            return 0
        eps = 1.e-6
        b0 = base.get_pos().view((-1,  1, 3)) # [n, 1, 3]
        b1 = base.get_pos().view(( 1, -1, 3)) # [1, n, 3]
        distance = (((b0 - b1)**2).sum(dim=2) + eps).sqrt() # [n, n]
        distance = torch.where(
            distance > eps**0.5,
            distance,
            torch.full((1,1), float('Inf'), device='cuda'))
        min_distance = distance.min(dim=0).values # [n,]
        return torch.std(min_distance) # [1,]

    def loss_all(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor, return_transients=False, transients=None):
        """[summary]

        Args:
            base (RbfBase): [description]
            transient_input (torch.Tensor): [description]
            return_transient (bool, optional): [description]. Defaults to False.

        Returns:
            [Tuple[Dict, torch.Tensor]]: Loss dictionary, full loss
        """
        weights = {
            'data': 1,
            # 'blob_size': 1,
            # 'blobs_distance': 1,
        }
        if transients is None:
            transients = self.base_to_transient(base, background_model, transient_input)
        losses = {
            'data': self.loss_data(transients[0], transient_input),
            # 'blob_size': 0 * self.loss_blob_size(base),
            # 'blobs_distance': 100 * self.loss_blob_distance(base),
        }
        full_loss = sum([
            weights[key] * losses[key]
            for key in losses.keys()])
        if return_transients:
            return losses, full_loss, transients
        return losses, full_loss

    # def optimize(self, base: RbfBase, transient_input: torch.Tensor,
    #              num_iter, base_lr=1.e-3):
    #     # Settings
    #     check_range = 10
    #     check_eps = 1.e-3
    #     # Stop if there is nothing to do
    #     if base.num_rbf() == 0:
    #         losses, full_loss = self.loss_all(base, transient_input)
    #         return losses, full_loss, 0
    #     base_backup = base.clone()
    #     for lr_reset in range(3):
    #         # Reduce lr with each retry
    #         lr = base_lr*0.5**lr_reset
    #         # Init Optimizer
    #         base.requires_grad(True)
    #         optim = torch.optim.Adam(base.param_list(), lr=lr)
    #         optim.zero_grad()
    #         # Track losses
    #         losses, full_loss = self.loss_all(base, transient_input)
    #         full_loss = full_loss
    #         if self.video_writer is not None:
    #             self.video_writer.write_frame(base, 'CHW', transform=self.render_visualization)
    #         loss_hist = [full_loss.item(),]
    #         loss_has_not_increased = False
    #         # Iterate
    #         for iteration in range(num_iter):
    #             full_loss.backward()
    #             optim.step()
    #             base.clamp_params()
    #             optim.zero_grad()
    #             if self.video_writer is not None and iteration % 2 == 0:
    #                 self.video_writer.write_frame(base, 'CHW', transform=self.render_visualization)
    #             losses, full_loss = self.loss_all(base, transient_input)
    #             loss_hist.append(full_loss.item())
    #             loss_has_not_increased = loss_hist[-1] <= loss_hist[0]
    #             if loss_has_not_increased and len(loss_hist) >= check_range:
    #                 rel_improv = (loss_hist[-check_range] - loss_hist[-1]) / loss_hist[-check_range]
    #                 if rel_improv < check_eps:
    #                     break
    #         base.requires_grad(False)
    #         # Check if ok
    #         if loss_has_not_increased:
    #             break
    #         # Reset
    #         base.params[:] = base_backup.params[:]
    #         print(f"WARNING: Optimization for lr={lr} failed.")
    #     if not loss_has_not_increased:
    #         print(f"WARNING: Optimization failed after {3} retries.")
    #     # detach
    #     full_loss = full_loss.detach()
    #     losses = {
    #         key: value.item() if isinstance(value, torch.Tensor) else value
    #         for key, value in losses.items()
    #     }
    #     return losses, full_loss, iteration + 1

    def optimize(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor,
                 num_iter, base_lr=1.e-3):
        # Settings
        loss_eps = 1.e-2
        difference_eps = 0.005
        lr = base_lr
        lr_multiplier = 0.5
        reset_counter = 0
        max_reset = 8
        # Stop if there is nothing to do
        if base.num_rbf() == 0:
            losses, full_loss = self.loss_all(base, background_model, transient_input)
            return losses, full_loss, 0
        base_backup = base.clone()
        background_model_backup = background_model.clone()
        # Reduce lr with each retry
        base.requires_grad(True)
        background_model.requires_grad(True)
        optim = torch.optim.Adam(base.param_list() + background_model.param_list(), lr=lr)
        # Track losses
        losses, full_loss, transients = self.loss_all(base, background_model, transient_input, return_transients=True)
        loss_multiplier = 1 / full_loss.detach()
        full_loss = full_loss * loss_multiplier
        last_loss = full_loss.detach()
        # Iterate
        iter_since_reset = 0
        for iteration in range(num_iter):
            # Step
            full_loss.backward()
            optim.step()
            base.clamp_params()
            background_model.clamp_parameters(transients[1], transients[2])
            optim.zero_grad()
            iter_since_reset += 1
            # Compute New loss
            losses, full_loss, transients = self.loss_all(base, background_model, transient_input, return_transients=True)
            full_loss = full_loss * loss_multiplier
            # Log current status
            if iteration % 2 == 0:
                if self.video_writer is not None:
                    self.video_writer.write_frame(base, 'CHW', transform=self.render_visualization)
            if iter_since_reset > 20 and iter_since_reset % 10 == 0:
                # Reset if loss has increased
                if full_loss > last_loss:
                    print(f'Loss has increased for lr {lr}')
                    if reset_counter == max_reset:
                        print(f"WARNING: Optimization failed after {reset_counter} retries.")
                        break
                    with torch.no_grad():
                        base.params[:] = base_backup.params[:]
                        background_model.copy_from(background_model_backup)
                    reset_counter += 1
                    lr = lr * lr_multiplier
                    optim = torch.optim.Adam(base.param_list() + background_model.param_list(), lr=lr)
                    optim.zero_grad()
                    losses, full_loss, transients = self.loss_all(base, background_model, transient_input, return_transients=True)
                    full_loss = full_loss * loss_multiplier
                    last_loss = full_loss.detach()
                    iter_since_reset = 0
                    continue
                # Check convergence
                if iter_since_reset >= 20:
                    rel_improv = (last_loss - full_loss.detach()) / last_loss
                    if rel_improv < loss_eps:
                        break
                # Compute difference and backup current iterate and loss
                with torch.no_grad():
                    # difference = base_backup.difference(base)
                    base_backup.params[:] = base.params[:]
                    background_model_backup.copy_from(background_model)
                    last_loss = full_loss.detach()
                # Check convergence
                # if difference < difference_eps:
                #     break
        # Reset
        # print(f'Optimization took {iteration+1} iterations with final lr {lr}.')
        base.requires_grad(False)
        background_model.requires_grad(False)
        full_loss = full_loss.detach() / loss_multiplier
        losses = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in losses.items()
        }
        return losses, full_loss, iteration + 1

    # def prune(self, base: RbfBase, transient_input: torch.Tensor, num_samples=5):
    #     if base.num_rbf() <= 1:
    #         return base, 0
    #     loss = self.loss_all(base, transient_input)[1]
    #     i = base.num_rbf() - num_samples
    #     while i < base.num_rbf():
    #         base_test = base.clone()
    #         base_test.remove(i)
    #         loss_test = self.optimize(base_test, transient_input, num_iter=50)[1]
    #         if loss_test <= loss:
    #             base = base_test
    #             loss = loss_test
    #         else:
    #             i += 1
    #         if self.video_writer is not None:
    #             self.video_writer.write_frame(base, 'CHW', transform=self.render_visualization)
    #     return base

    def add_blobs_volume(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor, num_samples=5):
        print('Add blobs volume: ', end='')
        # Compute loss
        _, initial_loss, transients = self.loss_all(base, background_model, transient_input, return_transients=True)
        # Backproject error into volume [D, H, W]
        volume_def = base.volume_def.clone()
        volume_def.resolution = tuple(r // 2 for r in volume_def.resolution)
        transient_error = (transient_input - transients[0]).clamp(min=0)
        if self.laser_points is None:
            volume_error = BackprojectionConfocalVelten.apply(
                transient_error[None],
                self.scan_points[None],
                volume_def,
                self.bin_def,
                None if self.scan_origin is None else self.scan_origin[None],
                ).clamp(min=0)
        else:
            volume_error = BackprojectionExhaustiveVelten.apply(
                transient_error[None],
                self.scan_points[None],
                self.laser_points[None],
                volume_def,
                self.bin_def,
                None if self.scan_origin is None else self.scan_origin[None],
                None if self.laser_origin is None else self.laser_origin[None],
                ).clamp(min=0)
        volume_error = second_derivative_filter(volume_error)
        volume_error = volume_error / volume_error.max()
        volume_error = volume_error**2
        # volume_error = volume_error + 1 / volume_error.numel()
        if self.summary_writer is not None:
            ve = volume_error / torch.max(volume_error)
            self.summary_writer.add_image('volume/max', ve[0].max(dim=0).values, self.run_var['iteration'], dataformats='HW')
            for i in range(ve.shape[1]):
                e = ve[0,i,:,:]
                self.summary_writer.add_image(f'volume/v{i}', e, self.run_var['iteration'], dataformats='HW')

        # Draw samples from flattened volume and add to base
        samples = torch.multinomial(volume_error.flatten(), num_samples, replacement=False).tolist()
        base_new = base.clone()
        background_model_new = background_model.clone()
        for sample in samples:
            x, y, z = volume_def.flat_idx_to_3d_idx(sample)
            x, y, z = volume_def.index_to_coordinates([x, y, z])
            z += 0.5 # base.rbf.half_distance()
            base_new.append(x, y, z)
        base_new.clamp_params()
        # Optimize
        _, add_loss = self.loss_all(base_new, background_model_new, transient_input)
        self.optimize(base_new, background_model_new, transient_input, num_iter=200)
        # Prune
        # base_new = self.prune(base_new, transient_input, num_samples)
        # Check success
        _, final_loss = self.loss_all(base_new, background_model_new, transient_input)
        success = final_loss <= initial_loss
        if success:
            print(f"Success! New loss={final_loss.item()} ({base.num_rbf()} to {base_new.num_rbf()} blobs) [init {initial_loss}, before optim {add_loss}]")
        else:
            print("Failed!")
            base_new = base
            background_model_new = background_model
        return base_new, background_model_new, success

    def sample_vertices(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor, num_samples):
        verts = base.make_verts()
        verts.requires_grad = True
        transients = self.render(verts, background_model, transient_input)
        loss = self.loss_all(base, background_model, transient_input, transients=transients)[1]
        loss.backward()
        verts_grad_length = (verts.grad**2).sum(dim=-1)**0.5
        if verts_grad_length.sum() == 0:
            return []
        verts = verts.detach()
        sample_indices = torch.multinomial(verts_grad_length, min(num_samples, verts.shape[0]), replacement=False).tolist()
        samples = []
        for idx in sample_indices:
            samples.append(verts[idx].tolist())
        return samples

    def add_blobs_surface(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor, num_samples=5):
        print('Add surface blobs: ', end='')
        # Compute loss
        _, initial_loss = self.loss_all(base, background_model, transient_input)
        # Sample vertices
        samples = self.sample_vertices(base, background_model, transient_input, num_samples)
        if len(samples) == 0:
            print('No samples...')
            return base, background_model, False
        base_new = base.clone()
        background_model_new = background_model.clone()
        for sample in samples:
            base_new.append(*sample[:3])
        base_new.clamp_params()
        # Optimize
        _, add_loss = self.loss_all(base_new, background_model_new, transient_input)
        self.optimize(base_new, background_model_new, transient_input, num_iter=200)
        # Prune
        # base_new = self.prune(base_new, transient_input, num_samples)
        # Check success
        _, final_loss = self.loss_all(base_new, background_model_new, transient_input)
        success = final_loss <= initial_loss
        if success:
            print(f"Success! New loss={final_loss.item()} ({base.num_rbf()} to {base_new.num_rbf()} blobs) [init {initial_loss}, before optim {add_loss}]")
        else:
            print("Failed!")
            base_new = base
            background_model_new = background_model
        return base_new, background_model_new, success

    def split_blobs(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor):
        if base.num_rbf() == 0:
            return base, False
        print('Split blobs: ', end='')
        # Compute loss
        _, initial_loss = self.loss_all(base, background_model, transient_input)
        num_blobs = base.num_rbf()
        new_blobs = min(min(32, num_blobs), max(0, 500 - num_blobs))
        # new_blobs = 32
        # Sample new blobs
        base_new = base.clone()
        background_model_new = background_model.clone()
        indices = torch.multinomial(base.get_sigma()[:,0], new_blobs, replacement=False).tolist()
        for idx in indices:
            # Reduce sigma
            base_new.params[idx, 3] = 0.75 * base_new.params[idx, 3]
            # Clone and add at the end
            base_new.params = torch.cat((base_new.params, base_new.params[None, idx]), dim=0)
            # Sample a movement direction and apply
            offset = base_new.params[idx,3] * (2 * torch.rand((2,), device='cuda') - 1)
            base_new.params[idx,:2] += offset
            base_new.params[ -1,:2] -= offset
        _, split_loss = self.loss_all(base_new, background_model_new, transient_input)
        _, final_loss, _ = self.optimize(base_new, background_model_new, transient_input, num_iter=400)
        success = final_loss <= initial_loss
        if success:
            print(f"Success! New loss={final_loss.item()} ({base.num_rbf()} to {base_new.num_rbf()} blobs) [init {initial_loss}, before optim {split_loss}]")
        else:
            print("Failed!")
            base_new = base
            background_model_new = background_model
        return base_new, background_model_new, success

    def check_delete(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor):
        if base.num_rbf() <= 1:
            return base, 0
        print('Check delete: ', end='')
        start_optimization_loss_increase = (self.delete_Factor - 1)*3 + 1
        blob_count_start = base.num_rbf()
        loss = self.loss_all(base, background_model, transient_input)[1]

        i = 0
        while i < base.num_rbf():
            base_test = base.clone()
            background_model_test = background_model.clone()
            base_test.remove(i)
            loss_test = self.loss_all(base_test, background_model_test, transient_input)[1]
            if self.optimize_check_delete and loss_test > loss*self.delete_Factor and loss_test <= loss*start_optimization_loss_increase:
                loss_test = self.optimize(base_test, background_model_test, transient_input, num_iter=50)[1]
            if loss_test <= loss*self.delete_Factor:
                base = base_test
                background_model = background_model_test
                loss = loss_test
            else:
                i += 1
            if self.video_writer is not None:
                self.video_writer.write_frame(base, 'CHW', transform=self.render_visualization)
        blob_count_end = base.num_rbf()
        blobs_removed = blob_count_start - blob_count_end
        if blobs_removed > 0:
            print(f"{blobs_removed} blobs removed. New data loss={loss.item()} ({blob_count_start} to {blob_count_end} blobs)")
        else:
            print("No blobs removed.")
        return base, background_model, blobs_removed

    def refine(self, base, background_model: TransientBackgroundBase, transient_input):
        print('Refine: ', end='')
        base_new = base.clone()
        background_model_new = background_model.clone()
        _, loss_before = self.loss_all(base, background_model, transient_input)
        _, loss_after, _ = self.optimize(base_new, background_model_new, transient_input, num_iter=500)
        success = loss_after < loss_before
        if success:
            print(f"Success! New loss={loss_after.item()} [init {loss_before}]")
        else:
            print("Failed!")
            base_new = base
            background_model_new = background_model
        return base_new, background_model_new, success

    def log(self, base: RbfBase, background_model: TransientBackgroundBase, transient_input: torch.Tensor, iteration: int):
        if self.summary_writer is None:
            return
        verts = base.make_verts()
        if base.has_color:
            colors = 255 * verts[None,...,3:4].expand(-1,-1,3).clamp(min=0, max=1)
            verts = verts[:,:3]
        else:
            colors = None
        faces = faces_of_verts(verts, None)
        losses, full_loss, transients = self.loss_all(base, background_model, transient_input, return_transients=True)
        transient_error = (transients[0] - transient_input).abs() / torch.max(torch.max(transient_input), torch.max(transients[0]))
        self.summary_writer.add_scalar('val/num_rbf', base.num_rbf(), iteration)
        self.summary_writer.add_scalar('loss/loss', full_loss, iteration)
        for key, value in losses.items():
            self.summary_writer.add_scalar(f'parts/{key}', value, iteration)
        if self.laser_points is None:
            self.summary_writer.add_image('img/transient', transients[0] / self.max_val, iteration, dataformats='HW')
            self.summary_writer.add_image('img/error', transient_error, iteration, dataformats='HW')
            if not isinstance(background_model, TransientBackgroundEmpty):
                self.summary_writer.add_image('img/transient_foreground', transients[1] / transients[1].max(), iteration, dataformats='HW')
                self.summary_writer.add_image('img/transient_background', transients[2] / transients[2].max(), iteration, dataformats='HW')
        else:
            self.summary_writer.add_image('img/transient', tensor_to_chw(transients[0] / self.max_val, 'HWN'), iteration, dataformats='CHW')
            self.summary_writer.add_image('img/error', tensor_to_chw(transient_error, 'HWN'), iteration, dataformats='CHW')
            if not isinstance(background_model, TransientBackgroundEmpty):
                self.summary_writer.add_image('img/transient_foreground', tensor_to_chw(transients[1] / transients[1].max(), 'HWN'), iteration, dataformats='CHW')
                self.summary_writer.add_image('img/transient_background', tensor_to_chw(transients[2] / transients[2].max(), 'HWN'), iteration, dataformats='CHW')
        self.summary_writer.add_mesh('mesh/mesh', verts_transform(verts[None]), faces=faces[None], colors=colors, global_step=iteration)
        if (base.num_rbf() > 0):
            self.summary_writer.add_histogram('blob/sigma', base.get_sigma(), iteration)
            self.summary_writer.add_histogram('blob/pos_x', base.get_pos()[:,0], iteration)
            self.summary_writer.add_histogram('blob/pos_y', base.get_pos()[:,1], iteration)
            self.summary_writer.add_histogram('blob/pos_z', base.get_pos()[:,2], iteration)
        self.summary_writer.flush()

    def fit(self,
            base: RbfBase,
            transient_input: torch.Tensor, # [T, S] or [T, S, L]
            background_model: TransientBackgroundBase = TransientBackgroundEmpty(),
            num_iter = 1000,
            verts_input: typing.Optional[torch.Tensor] = None,
            faces_input: typing.Optional[torch.Tensor] = None,
            colors_input: typing.Optional[torch.Tensor] = None,
            log_interval = 1,
            subdivide_at = [],
            callback = None) -> RbfBase:
        # Reduce empty transient
        self.bin_def = self.bin_def_init
        # transient_input = self.reduce_transient(transient_input)
        # Normalize
        transient_input = transient_input / transient_input.mean()
        self.max_val = transient_input.max()
        self.run_var = {}
        # Init Summary writer, video writer and write inputs
        if self.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(
                log_dir=self.log_dir
            )
            self.summary_writer.add_image(
                'img/reference',
                tensor_to_chw(transient_input / self.max_val, 'HWN'),
                dataformats='CHW')
            if verts_input is not None:
                self.summary_writer.add_mesh(
                    'mesh/input',
                    verts_transform(verts_input[None]),
                    faces=faces_input[None] if faces_input is not None else None,
                    colors=255*colors_input[None].expand(-1,-1,3) if colors_input is not None else None)
        else:
            self.summary_writer = None
        if self.video:
            self.video_writer = RealTimeVideoWriter(
                os.path.join(self.summary_writer.log_dir, 'test.mp4'),
                (512, 512), speed=1.0)
        else:
            self.video_writer = None

        # Init optimization
        try_add_next = True
        blobs_removed = 0

        # Global iteration
        try:
            for i in range(num_iter):
                self.run_var['iteration'] = i
                print(f'\nIteration {i}/{num_iter}:')
                if i%5 == 0 or base.num_rbf() == 0:
                    base, background_model, success_1 = self.add_blobs_volume(base, background_model, transient_input, 5)
                elif i%5 == 1:
                    base, background_model, success_2 = self.add_blobs_surface(base, background_model, transient_input, 5)
                elif i%5 == 2:
                    base, background_model, success_3 = self.split_blobs(base, background_model, transient_input)
                elif i%5 == 3:
                    base, background_model, blobs_removed = self.check_delete(base, background_model, transient_input)
                elif i%5 == 4:
                    base, background_model, refine_success = self.refine(base, background_model, transient_input)

                if i % log_interval == 0:
                    self.log(base, background_model, transient_input, i)
                
                if i in subdivide_at:
                    base.subdivide()
                
                if callback is not None:
                    callback(i, base, background_model)

        finally:
            if self.video_writer is not None:
                self.video_writer.release()
        return base
