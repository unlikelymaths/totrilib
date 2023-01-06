import typing
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer, FoVOrthographicCameras, RasterizationSettings

class UnitDepthMapRender():

    def __init__(self, resolution=512, xysize=2, znear=0, zfar=2, device=torch.device('cpu')):
        cameras = FoVOrthographicCameras(
            znear=znear, zfar=zfar,
            min_x=-xysize/2, max_x=xysize/2,
            min_y=-xysize/2, max_y=xysize/2,
            device=device)
        raster_settings = RasterizationSettings(resolution, perspective_correct=True, cull_backfaces=False)
        self.rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )

    def apply(self, verts: torch.Tensor, faces: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        """_summary_

        Args:
            verts (torch.Tensor): vertices [V, 3]
            faces (torch.Tensor): faces [F, 3]

        Returns:
            (torch.Tensor, torch.Tensor): depth map [H, W], mask [H, W]
        """
        mesh = Meshes([verts,], [faces,])
        fragments = self.rasterizer(mesh)
        depth = fragments.zbuf[0,:,:,0]
        mask = fragments.pix_to_face[0,:,:,0] > -1
        return depth, mask
