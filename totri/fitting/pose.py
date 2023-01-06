import copy
import typing
import torch
from pytorch3d.transforms import quaternion_apply
from totri.types import BinDef
from totri.render import MeshRenderConfocal
from totri.data.util import merge_meshes
from totri.util.render import UnitCubeMeshRender

class Translation(torch.nn.Module):

    def __init__(self, translation=(0, 0, 0), device='cuda'):
        super().__init__()
        translation = torch.tensor(
            [translation], dtype=torch.float32, device=device)
        self.translation = torch.nn.Parameter(translation)

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            verts (torch.Tensor): verts [N, 3]

        Returns:
            torch.Tensor: translated verts [N, 3]
        """
        return verts + self.translation

class Rotation(torch.nn.Module):

    def __init__(self, quaternion=(1, 0, 0, 0), device='cuda'):
        super().__init__()
        quaternion = torch.tensor(
            [quaternion], dtype=torch.float32, device=device)
        self.quaternion = torch.nn.Parameter(quaternion)

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            verts (torch.Tensor): verts [N, 3]

        Returns:
            torch.Tensor: translated verts [N, 3]
        """
        quaternion_normalized = self.quaternion / (((self.quaternion**2).sum()+1.e-8)**0.5)
        return quaternion_apply(quaternion_normalized, verts)

def apply_transform(verts_list, translation_list, rotation_list):
    return [
        translation_list[i](rotation_list[i](verts_list[i]))
        for i in range(len(verts_list))]

class PoseFitting():

    def __init__(self,
                 bin_def: BinDef,
                 tensorboard: bool,
                 log_dir: str = None):
        self.bin_def = bin_def
        self.tensorboard = tensorboard
        self.log_dir = log_dir
        self.summary_writer = None
        self.losses = []

    def fit(self, transient_input: torch.Tensor, scan_points: torch.Tensor,
            verts_list: typing.List[torch.Tensor],
            faces_list: typing.List[torch.Tensor],
            verts_gt_list: typing.List[torch.Tensor] = None,
            translation_list = None,
            rotation_list = None,
            num_iter = 500):
        """[summary]

        Args:
            transient_input (torch.Tensor): [T, S]
            scan_points (torch.Tensor): [S, 3]
            verts (torch.Tensor): [V, 3]
            faces (torch.Tensor): [F, 3]
        """
        if self.tensorboard and self.summary_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(
                log_dir=self.log_dir
            )
        
        num_meshes = len(verts_list)
        max_val = torch.max(transient_input)
        transient_input = transient_input / max_val
        if translation_list is None:
            translation_list = [Translation() for i in range(num_meshes)]
        translation_parameters = []
        for i in range(num_meshes):
            translation_parameters += list(translation_list[i].parameters())
        if rotation_list is None:
            rotation_list = [Rotation() for i in range(num_meshes)]
        rotation_parameters = []
        for i in range(num_meshes):
            rotation_parameters += list(rotation_list[i].parameters())
        if self.summary_writer is not None:
            self.summary_writer.add_image(
                'img/reference', 
                transient_input,
                dataformats='HW')
            verts_merged, faces_merged = merge_meshes(apply_transform(verts_list, translation_list, rotation_list), faces_list)
            self.summary_writer.add_image(
                'img/initial_pose', 
                UnitCubeMeshRender().apply(verts_merged, faces_merged),
                dataformats='CHW')
            if verts_gt_list is not None:
                verts_merged, faces_merged = merge_meshes(verts_gt_list, faces_list)
                self.summary_writer.add_image(
                    'img/target_pose', 
                    UnitCubeMeshRender().apply(verts_merged, faces_merged),
                    dataformats='CHW')

        optim = torch.optim.Adam(
            translation_parameters + rotation_parameters,
            lr=1.e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, [50, 100, 200, 400], gamma=0.8)

        for i in range(num_iter):
            optim.zero_grad()

            verts_merged, faces_merged = merge_meshes(apply_transform(verts_list, translation_list, rotation_list), faces_list)
            transient = MeshRenderConfocal.apply(
                    verts_merged[None], faces_merged[None], scan_points[None],
                    self.bin_def, None, None)[0] / max_val
            loss = 100 * torch.mean((transient - transient_input)**2)
            loss.backward()
            self.losses.append(loss.detach())

            optim.step()
            scheduler.step()

            if i%10 == 0:
                print(f'{i}/{num_iter}: {loss.item()}')

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('loss/loss', loss, i)
                self.summary_writer.add_image(
                    'img/transient', 
                    transient / torch.max(transient),
                    i, dataformats='HW')
                self.summary_writer.add_image(
                    'img/error', 
                    (transient - transient_input).abs() / torch.max(transient, transient_input).max(),
                    i, dataformats='HW')
                self.summary_writer.add_image(
                    'img/current_pose', 
                    UnitCubeMeshRender().apply(verts_merged, faces_merged),
                    i, dataformats='CHW')
            if loss < 1.e-6:
                print(f'Finished after {i+1} iterations')
                break
        if self.summary_writer is not None:
            verts_merged, faces_merged = merge_meshes(apply_transform(verts_list, translation_list, rotation_list), faces_list)
            self.summary_writer.add_image(
                'img/final_pose', 
                UnitCubeMeshRender().apply(verts_merged, faces_merged),
                dataformats='CHW')
        return translation_list, rotation_list

class PoseVideoFitting():

    def __init__(self,
                 bin_def: BinDef):
        self.bin_def = bin_def

    def fit(self, transient_input_list: torch.Tensor,
            scan_points: torch.Tensor,
            verts_list: typing.List[torch.Tensor],
            faces_list: typing.List[torch.Tensor],
            translation_list_0,
            rotation_list_0,
            num_iter = 500):
        """[summary]

        Args:
            transient_input (torch.Tensor): [T, S]
            scan_points (torch.Tensor): [S, 3]
            verts (torch.Tensor): [V, 3]
            faces (torch.Tensor): [F, 3]
        """
        num_frames = len(transient_input_list)
        translation_lists = [translation_list_0,]
        rotation_lists = [rotation_list_0,]
        for i in range(num_frames):
            fitting = PoseFitting(self.bin_def, False)
            translation_lists[i], rotation_lists[i] = fitting.fit(
                transient_input_list[i],
                scan_points,
                verts_list, faces_list,
                translation_list = translation_lists[i],
                rotation_list = rotation_lists[i],
                num_iter = num_iter
            )
            if i < num_frames - 1:
                translation_lists.append(copy.deepcopy(translation_lists[i]))
                rotation_lists.append(copy.deepcopy(rotation_lists[i]))
        return translation_lists, rotation_lists
