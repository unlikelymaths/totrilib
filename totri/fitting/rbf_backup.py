import os
import abc
import torch
from tomcubes import MarchingCubesFunction
from totri.types import C0, BinDef, VolumeDef
from totri.render import MeshRenderConfocal
from totri.util import RealTimeVideoWriter
from totri.util.render import UnitCubeMeshRender
from totri.util.format import verts_transform, faces_of_verts
from totri.reconstruct import (
    BackprojectionConfocalVelten, second_derivative_filter)



# Util
def normalize(t):
    return t / torch.max(t)

class RadialBasisFunction(abc.ABC):

    def __init__(self, num_channels: int, has_color: bool):
        self.num_channels = num_channels
        self.has_color = has_color

    @abc.abstractmethod
    def make_volume(self, blobs: torch.Tensor, coords: torch.Tensor,
                    volume_mask: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def append(self, blobs: torch.Tensor, x: float, y: float, z:float) -> torch.Tensor:
        pass

class GaussianRbf(RadialBasisFunction):

    def __init__(self, has_color: bool, sigma_init: float = 0.05, color_init: float = 1.0):
        num_channels = 4
        if has_color:
            num_channels += 1
        super().__init__(num_channels, has_color)
        self.sigma_init = sigma_init
        self.color_init = color_init

    def make_volume(self, blobs: torch.Tensor, coords: torch.Tensor,
                    volume_mask: torch.Tensor) -> torch.Tensor:
        # blobs [N, 4+A]
        # coords [D, H, W, 3]
        # volume_mask [D, H, W]
        # return [1+A, D, H, W]
        volume = torch.zeros_like(coords[...,0])
        if self.has_color:
            color = torch.zeros_like(coords[...,0])
        for i in range(blobs.shape[0]):
            weight = torch.exp(-torch.sum(
                (coords - blobs[i,:3].view((1, 1, 1, 3)))**2 / (2 * blobs[i,3:4]**2),
                dim=3))
            volume = volume + weight
            if self.has_color:
                color = color + blobs[i, 4] * weight
        if self.has_color:
            color = color / (volume + 1.e-12)
        volume = volume * volume_mask
        if self.has_color:
            return torch.stack((volume, color), dim=0)
        else:
            return volume[None]

    def append(self, blobs: torch.Tensor, x: float, y: float, z:float) -> torch.Tensor:
        if self.has_color:
            new_blob = torch.tensor(
                [[x, y, z, self.sigma_init, self.color_init]],
                dtype=torch.float32, device=blobs.device)
        else:
            new_blob  = torch.tensor(
                [[x, y, z, self.sigma_init]],
                dtype=torch.float32, device=blobs.device)
        return torch.cat((blobs, new_blob), dim=0)


class InverseMultiquadraticRbf():
    pass

class RbfBase():
    def __init__(self, rbf: RadialBasisFunction, volume_def: VolumeDef,
                 blobs=None, coords=None, volume_mask=None):
        self.rbf = rbf
        self.volume_def = volume_def
        self.blobs = blobs if blobs is not None else torch.empty((0, self.rbf.num_channels), device='cuda')
        """Blobs [N, 4(+A)] with entries [x, y, z, sigma, (attributes)]"""
        self.coords = coords if coords is not None else volume_def.coordinates_inclusive(device='cuda')
        """Voxel coordinates [D, H, W, 3]"""
        self.volume_mask = volume_mask if volume_mask is not None else self.get_volume_mask()
        """Binary volume mask [D, H, W]"""

    def get_volume_mask(self):
        volume_mask = torch.ones_like(self.coords[:,:,:,0])
        volume_mask[ 0,:,:] = 0
        volume_mask[-1,:,:] = 0
        volume_mask[:, 0,:] = 0
        volume_mask[:,-1,:] = 0
        volume_mask[:,:, 0] = 0
        volume_mask[:,:,-1] = 0 
        return volume_mask

    def clone(self):
        return RbfBase(self.rbf, self.volume_def, self.blobs.clone(),
                       self.coords, self.volume_mask)

    def requires_grad(self, requires_grad: bool):
        self.blobs.requires_grad = requires_grad

    def params(self):
        return [self.blobs]

    def append(self, x, y, z):
        self.blobs = self.rbf.append(self.blobs, x, y, z)

    def remove(self, idx):
        self.blobs = torch.cat(
            (self.blobs[:idx], self.blobs[idx+1:]), dim=0)

    def make_volume(self) -> torch.Tensor:
        # return [1+A, D, H, W]
        return self.rbf.make_volume(self.blobs, self.coords, self.volume_mask)

    def num_rbf(self):
        return self.blobs.shape[0]

    def get_pos(self):
        # pos [N, 3]
        return self.blobs[:,:3]

    def get_sigma(self):
        # pos [N, 1]
        return self.blobs[:,3:4]

class RbfFitting():

    def __init__(self, rbf: RadialBasisFunction, bin_def: BinDef, volume_def: VolumeDef,
                 scan_points: torch.Tensor, sigma_target: float = 0.02,
                 sigma_min: float = 0.001,
                 margin: float = 0.1, tensorboard: bool = False):
        self.rbf = rbf
        self.bin_def = bin_def
        self.volume_def = volume_def
        self.scan_points = scan_points
        """Scan Points [S, 3]"""
        self.sigma_target = sigma_target
        self.sigma_min = sigma_min
        self.margin = margin
        self.tensorboard = tensorboard
        self.ucmr = UnitCubeMeshRender()

    def marching_cubes(self, volume) -> torch.Tensor:
        # volume [1+A, D, H, W]
        # return verts [t, 3+A]
        verts = MarchingCubesFunction.apply(
            volume[None],
            0.1,
            self.volume_def.start, self.volume_def.end)[0]
        return verts

    def render_visualization(self, base: RbfBase):
        # -> [3, H, W]
        with torch.no_grad():
            volume = base.make_volume()
            verts = self.marching_cubes(volume)
            if self.rbf.has_color:
                colors = verts[:,3:4]
            else:
                colors = None
            return self.ucmr.apply(verts[:,:3], colors=colors)

    def render(self, verts):
        # verts [t, 3+A] -> [s, b]
        if self.rbf.has_color:
            color = torch.clamp(verts[None, :, 3:4].contiguous(), min=1.e-6)
        else:
            color = None
        verts = verts[None, :, :3].contiguous()
        transient = MeshRenderConfocal.apply(
            verts, None,
            self.scan_points[None],
            self.bin_def,
            color, None)
        return transient[0]

    def base_to_transient(self, base: RbfBase):
        # RbfBase -> [s, b]
        volume = base.make_volume()
        verts = self.marching_cubes(volume)
        transient = self.render(verts)
        return transient

    def loss_data(self, transient: torch.Tensor, transient_input: torch.Tensor):
        # [s, b], [s, b] -> [1,]
        return 1.e4 * torch.mean((transient - transient_input)**2)

    def loss_blob_size(self, base: RbfBase):
        # RbfBase -> [1,]
        if base.num_rbf() < 1:
            return 0
        sigma = base.get_sigma()
        return torch.sum((sigma - self.sigma_target)**2)

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

    def loss_all(self, base: RbfBase, transient_input: torch.Tensor, return_transient=False):
        weights = {
            'data': 1,
            'blob_size': 1,
            'blobs_distance': 1,
        }
        transient = self.base_to_transient(base)
        losses = {
            'data': self.loss_data(transient, transient_input),
            'blob_size': 1000 * self.loss_blob_size(base),
            'blobs_distance': 1000 * self.loss_blob_distance(base),
        }
        full_loss = sum([
            weights[key] * losses[key]
            for key in losses.keys()])
        if return_transient:
            return losses, full_loss, transient
        return losses, full_loss

    def clamp_blob_parameters(self, base: RbfBase):
        with torch.no_grad():
            pos = base.get_pos()
            for x in range(3):
                pos[:,x] = pos[:,x].clamp(
                    self.volume_def.start[x] + self.margin,
                    self.volume_def.end[x]   - self.margin)
            sigma = base.get_sigma()
            sigma[:] = sigma[:].clamp(min=self.sigma_min)

    def optimize(self, base: RbfBase, transient_input: torch.Tensor,
                 num_iter, base_lr=1.e-3):
        # Settings
        check_range = 10
        check_eps = 1.e-3
        # Stop if there is nothing to do
        if base.num_rbf() == 0:
            losses, full_loss = self.loss_all(base, transient_input)
            return losses, full_loss, 0
        base_backup = base.clone()
        for lr_reset in range(3):
            # Reduce lr with each retry
            lr = base_lr*0.5**lr_reset
            # Init Optimizer
            base.requires_grad(True)
            optim = torch.optim.Adam(base.params(), lr=lr)
            optim.zero_grad()
            # Track losses
            losses, full_loss = self.loss_all(base, transient_input)
            if self.video_writer is not None:
                self.video_writer.write_frame(base, 'CHW', transform=self.render_visualization)
            loss_hist = [full_loss.item(),]
            loss_has_not_increased = False
            # Iterate
            for iteration in range(num_iter):
                full_loss.backward()
                optim.step()
                self.clamp_blob_parameters(base)
                optim.zero_grad()
                losses, full_loss = self.loss_all(base, transient_input)
                if self.video_writer is not None and iteration % 2 == 0:
                    self.video_writer.write_frame(base, 'CHW', transform=self.render_visualization)
                loss_hist.append(full_loss.item())
                loss_has_not_increased = loss_hist[-1] <= loss_hist[0]
                if loss_has_not_increased and len(loss_hist) >= check_range:
                    rel_improv = (loss_hist[-check_range] - loss_hist[-1]) / loss_hist[-check_range]
                    if rel_improv < check_eps:
                        break
            base.requires_grad(False)
            # Check if ok
            if loss_has_not_increased:
                break
            # Reset
            base.blobs[:] = base_backup.blobs[:]
            print(f"WARNING: Optimization for lr={lr} failed.")
        if not loss_has_not_increased:
            print(f"WARNING: Optimization failed after {3} retries.")
        # detach
        full_loss = full_loss.detach()
        losses = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in losses.items()
        }
        return losses, full_loss, iteration + 1

    def add_best_blob(self, base: RbfBase, transient_input: torch.Tensor, num_samples=5):
        print('Add best blob: ', end='')
        # Compute current loss
        _, initial_loss, transient = self.loss_all(base, transient_input, return_transient=True)
        # Backproject error into volume [D, H, W]
        transient_error = (transient_input - transient)
        volume_error = BackprojectionConfocalVelten.apply(
            transient_error[None], self.scan_points[None],
            self.volume_def, self.bin_def).clamp(min=0)
        volume_error = second_derivative_filter(volume_error)
        volume_error = volume_error / volume_error.max()
        volume_error = volume_error**2
        # Draw samples from flattened volume
        samples = torch.multinomial(volume_error.flatten(), num_samples, replacement=False).tolist()
        # Find best new blob
        loss_best = initial_loss
        base_best = base
        for sample in samples:
            x, y, z = self.volume_def.flat_idx_to_3d_idx(sample)
            x, y, z = self.volume_def.index_to_coordinates([x, y, z])
            base_new = base.clone()
            base_new.append(x, y, z)
            self.clamp_blob_parameters(base_new)
            _, loss_new, _ = self.optimize(base_new, transient_input, num_iter=200)
            if loss_new <= loss_best:
                base_best = base_new
                loss_best = loss_new
        success = loss_best < initial_loss
        if success:
            print(f"Success! New loss={loss_best.item()} ({base.num_rbf()} to {base_best.num_rbf()} blobs)")
        else:
            print("Failed!")
        return base_best, success

    def add_surface_blobs(self, base: RbfBase, transient_input: torch.Tensor, num_samples=10):
        print('Add surface blobs: ', end='')
        # Compute current loss
        _, loss_before = self.loss_all(base, transient_input)
        # Get verts
        verts = self.marching_cubes(base.make_volume())[:,:3]
        # Draw samples from vertices
        samples = torch.randperm(verts.shape[0])[:num_samples].tolist()
        # Add blobs
        base_new = base.clone()
        for idx in samples:
            base_new.append(verts[idx,0], verts[idx,1], verts[idx,2])
        self.clamp_blob_parameters(base_new)
        _, loss_after, _ = self.optimize(base_new, transient_input, num_iter=200)
        success = loss_after < loss_before
        if success:
            print(f"Success! New loss={loss_after.item()} ({base.num_rbf()} to {base_new.num_rbf()} blobs)")
        else:
            print("Failed!")
            base_new = base
        return base_new, success

    # def split_blobs(blobs):
    #     print('Split blobs: ', end='')
    #     num_blobs = blobs.shape[0]
    #     max_blobs = 500
    #     if num_blobs >= max_blobs:
    #         print(f'Maximum number of blobs reached ({max_blobs})')
    #     new_blobs = min(min(32, 8*num_blobs), max_blobs-num_blobs)
    #     # Sample new blobs
    #     blobs_backup = blobs.clone()
    #     for i in range(new_blobs):
    #         # Sample a random blob
    #         idx = torch.randint(0, blobs.shape[0], (1,)).item()
    #         # Reduce sigma
    #         blobs[idx,3] = torch.clamp(blobs[idx,3]/2, min=sig_min)
    #         # Clone and add at the end
    #         blobs = torch.cat((blobs, blobs[None, idx]), dim=0)
    #         # Sample a movement direction and apply
    #         offset = blobs[idx,3] * torch.randn((3,), device='cuda')
    #         blobs[idx,:3] += offset
    #         blobs[ -1,:3] -= offset
    #     _, _, _ = optimize(blobs, 400)
    #     with torch.no_grad():
    #         _, loss0 = loss_all(blobs_backup, transient_input)
    #         _, loss1 = loss_all(blobs,        transient_input)
    #     success = loss1 < loss0
    #     if success:
    #         print(f"Success! New loss={loss1.item()} ({blobs_backup.shape[0]} to {blobs.shape[0]} blobs)")
    #     else:
    #         print("Failed!")
    #         blobs = blobs_backup
    #     return blobs, success

    def check_delete(self, base: RbfBase, transient_input: torch.Tensor):
        if base.num_rbf() <= 1:
            return base, 0
        print('Check delete: ', end='')
        maximum_relative_loss_increase = 1.001
        blob_count_start = base.num_rbf()
        loss = self.loss_all(base, transient_input)[0]['data']

        i = 0
        while i < base.num_rbf():
            base_test = base.clone()
            base_test.remove(i)
            loss_test = self.loss_all(base_test, transient_input)[0]['data']
            if loss_test <= loss*maximum_relative_loss_increase:
                base = base_test
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
        return base, blobs_removed

    def refine(self, base, transient_input):
        print('Refine: ', end='')
        base_new = base.clone()
        _, loss_before = self.loss_all(base, transient_input)
        _, loss_after, _ = self.optimize(base_new, transient_input, num_iter=500)
        success = loss_after < loss_before
        if success:
            print(f"Success! New loss={loss_after.item()}")
        else:
            print("Failed!")
            base_new = base
        return base_new, success

    def log(self, base: RbfBase, transient_input: torch.Tensor, iteration: int):
        if self.summary_writer is None:
            return
        volume = base.make_volume()
        verts = self.marching_cubes(volume)
        faces = faces_of_verts(verts, None)
        losses, full_loss, transient = self.loss_all(base, transient_input, return_transient=True)
        transient_error = (transient - transient_input).abs() / torch.max(torch.max(transient_input), torch.max(transient))
        self.summary_writer.add_scalar('val/num_rbf', base.num_rbf(), iteration)
        self.summary_writer.add_scalar('loss/loss', full_loss, iteration)
        for key, value in losses.items():
            self.summary_writer.add_scalar(f'parts/{key}', value, iteration)
        self.summary_writer.add_image('img/transient', normalize(transient), iteration, dataformats='HW')
        self.summary_writer.add_image('img/error', transient_error, iteration, dataformats='HW')
        self.summary_writer.add_mesh('mesh/mesh', verts_transform(verts[None]), faces=faces[None], global_step=iteration)
        if (base.num_rbf() > 0):
            self.summary_writer.add_histogram('blob/sigma', base.get_sigma(), iteration)
        self.summary_writer.flush()

    def fit(self, transient_input: torch.Tensor,
            verts_input: torch.Tensor, faces_input: torch.Tensor) -> RbfBase:
        # Init Summary writer, video writer and write inputs
        if self.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter()
            self.summary_writer.add_image(
                'img/reference',
                normalize(transient_input),
                dataformats='HW')
            if verts_input is not None:
                self.summary_writer.add_mesh(
                    'mesh/input',
                    verts_transform(verts_input[None]), faces=faces_input[None])
            self.video_writer = RealTimeVideoWriter(
                os.path.join(self.summary_writer.log_dir, 'test.mp4'),
                (512, 512), speed=1)
        else:
            self.summary_writer = None
            self.video_writer = None

        # Init optimization
        base = RbfBase(self.rbf, self.volume_def)
        num_iter = 20
        last_add_success = False
        blobs_removed = 0

        # Global iteration
        try:
            for i in range(num_iter):
                print(f'\nIteration {i}/{num_iter}:')
                # Try to add a new blob if there are no blobs or the last add failed
                if base.num_rbf() < 1 or not last_add_success:
                    base, last_add_success = self.add_best_blob(base, transient_input)
                else:
                    base, last_add_success = self.add_surface_blobs(base, transient_input, 10)

                # Check delete only if something was added, or every 10 iterations
                if last_add_success or i+1 % 10 == 0:
                    base, blobs_removed = self.check_delete(base, transient_input)
                else:
                    blobs_removed = 0

                # Refine only if add or remove changed something
                if last_add_success or blobs_removed > 0 or i+1%5 == 0:
                    base, refine_success = self.refine(base, transient_input)

                self.log(base, transient_input, i)
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
        return base




from totri.data.util import make_wall_grid
from totri.data.mesh import load_sample_mesh

bin_def=BinDef(2*2/512, 0.5, 512, unit_is_time=False)
resolution=32
volume_def=VolumeDef([-1, -1, 0], [1, 1, 2], [resolution,]*3)
scan_points = make_wall_grid(
    -1, 1, 32,
    -1, 1, 32,
    z_val=0)
rbf = GaussianRbf(has_color=False)


verts, faces = load_sample_mesh('bunny')
transient = MeshRenderConfocal.apply(
        verts, faces, scan_points[None],
        bin_def, None, None)[0]


base = RbfFitting(
    rbf, bin_def, volume_def, scan_points, tensorboard=True
).fit(
    transient,
    verts[0], faces[0]
)
