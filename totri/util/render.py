import typing
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    PointLights, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)

from totri.types import VolumeDef

class UnitCubeMeshRender():

    def __init__(self, resolution=512, distance=0.5, specular=0.2, eye=None):
        if eye is None:
            eye = [0.0, 0.0, -distance]
        R, T = look_at_view_transform(
            eye=[eye],
            up=[[0.0, 1.0, 0.0]],
            at=[[0.0, 0.0, 1.0]],
            device='cuda') 
        cameras = FoVPerspectiveCameras(R=R, T=T, device='cuda', znear=0.1)
        raster_settings = RasterizationSettings(
            image_size=resolution, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        lights = PointLights(
            location=[[2.0, 2.0, -2.0]],
            specular_color=[[specular, specular, specular]],
            device='cuda')
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device='cuda', 
                cameras=cameras,
                lights=lights
            )
        )

    def apply(self, verts, faces=None, colors=None,
              volume_def: typing.Optional[VolumeDef] = None):
        """[summary]

        Args:
            verts: [V, 3]
            faces: [F, 3]
            colors: [V, 3]
        
        Returns:
            Image [3, H, W]
        """
        if volume_def is not None:
            start = [min(x, y) for x, y in zip(volume_def.start, volume_def.end)]
            sizes = [abs(x - y) for x, y in zip(volume_def.start, volume_def.end)]
            max_size = max(sizes)
            verts = torch.stack((
                (verts[:,0] - start[0]) / max_size * 2 - sizes[0] / max_size,
                (verts[:,1] - start[1]) / max_size * 2 - sizes[1] / max_size,
                (verts[:,2] - start[2]) / max_size * 2 + (1 - 0.5 * sizes[2] / max_size),
                ), dim=1)
        if faces is None:
            faces = torch.arange(
                0, verts.shape[0],
                dtype=torch.int32, device=verts.device).view(-1, 3)
        if colors is None:
            colors = torch.tensor(
                [[1,1,1]],
                dtype=torch.float32,
                device=verts.device) # (1, 3)
        mesh = Meshes([verts,], [faces,])
        mesh.textures = TexturesVertex(verts_features=[colors.expand(verts.shape[0], -1),])
        return self.renderer(mesh)[0,:,:,:3].permute(2, 0, 1)
