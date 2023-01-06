"""Fk Migration"""
import torch
import torch.fft

class StoltInterpolation(torch.nn.Module):

    def __init__(self, depth, height, width):
        """Initialize with depth, height and width of the reconstruction volume

        depth:  also range, number of bins times bin resolution times c [meter]
        height: height of the scanning points on the wall [meter]
        width:  width of the scanning points on the wall [meter]
        """
        super().__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.last_shape = (None, None, None) # D, H, W
        self.buffers = [None, None]

    def get_buffers(self, D, H, W, device):
        """Initialize with depth, height and width of the reconstruction volume

        D:      Number of temporal bins
        H:      Number of scanning points in vertical direction
        W:      Number of scanning points in horizontal direction
        device: Device for the tensors

        Returns
          grid: 1, 2*D, H, W, 3
          weights: 1, 2*D, H, W
        """
        if self.buffers[0] is not None:
            if (D == self.last_shape[0] and
                H == self.last_shape[1] and
                W == self.last_shape[2]):
                self.buffers[0] = self.buffers[0].to(device)
                self.buffers[1] = self.buffers[1].to(device)
                return self.buffers
        # Make Grid
        x = torch.linspace(-1, 1, W,   device=device).view(1,   1, W).expand((2*D, H, W))
        y = torch.linspace(-1, 1, H,   device=device).view(1,   H, 1).expand((2*D, H, W))
        z = torch.linspace(-1, 1, 2*D, device=device).view(2*D, 1, 1).expand((2*D, H, W))
        zz = torch.sqrt(
            x**2 * ((W*self.depth) / (2*D*self.width ))**2 +
            y**2 * ((H*self.depth) / (2*D*self.height))**2 +
            z**2)
        grid = torch.stack((x, y, zz), dim=3).unsqueeze(0)
        weights = (z > 0) * torch.abs(z) / torch.clamp(zz, min=1e-6)
        weights = weights.unsqueeze(0)
        self.buffers[0] = grid
        self.buffers[1] = weights
        self.last_shape = (D, H, W)
        return self.buffers

    def forward(self, fourier : torch.Tensor):
        """Apply Stolt Interpolation to the volume
        
        fourier: complex valued volume of shape (N C 2D H W)
                 N batchsize
                 C channels
                 D Number of temporal bins
                 H Number of scanning points in vertical direction
                 W Number of scanning points in horizontal direction
        """
        # Get Buffers
        N, C, D2, H, W = fourier.shape
        assert(D2%2 == 0)
        D = D2//2
        grid, weights = self.get_buffers(D, H, W, fourier.device)
        # Interpolate
        fourier = fourier.view(N*C, D2, H, W)
        fourier = torch.view_as_real(fourier) # real NC 2*D H W 2
        fourier_real = torch.nn.functional.grid_sample(
            fourier[None, :, :, :, :, 0], # 1, NC, 2*D, H, W
            grid, # 1, 2*D, H, W, 3
            mode="bilinear", padding_mode="zeros", align_corners=False).view(N*C, D2, H, W)
        fourier_imag = torch.nn.functional.grid_sample(
            fourier[None, :, :, :, :, 1], # 1, NC, 2*D, H, W
            grid, # 1, 2*D, H, W, 3
            mode="bilinear", padding_mode="zeros", align_corners=False).view(N*C, D2, H, W)
        fourier = torch.view_as_complex(torch.stack((fourier_real, fourier_imag), dim=4)) # NC, 2*D, H, W
        # Weight
        fourier = fourier * weights
        return fourier.view(N, C, D2, H, W)

class FkMigration(torch.nn.Module):

    def __init__(self, depth, height, width):
        super().__init__()
        self.stolt_interpolation = StoltInterpolation(depth, height, width)

    def forward(self, transient: torch.Tensor) -> torch.Tensor:
        """Apply Fk migration

        Args:
            transient (torch.Tensor): transient volume of shape [N, D, H, W]
                N batchsize
                D Number of temporal bins
                H Number of scanning points in vertical direction
                W Number of scanning points in horizontal direction

        Returns:
            torch.Tensor: Volume [N, D, H, W]
        """
        device = transient.device
        D = transient.shape[1]
        transient = transient.unsqueeze(1)
        # Step 0: Pad data
        grid_z = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        transient = torch.sqrt(transient) * grid_z
        transient = torch.cat((transient, torch.zeros_like(transient)), dim=2)
        # Step 1: FFT
        fourier = torch.fft.fftshift(torch.fft.fftn(transient, dim=(2, 3, 4)), dim=(2, 3, 4)) # complex N 2*D H W
        # Step 2: Stolt trick
        fourier = self.stolt_interpolation(fourier) # complex N 2*D H W
        # Step 3: IFFT
        volume = torch.fft.ifftn(torch.fft.ifftshift(fourier, dim=(2, 3, 4)), dim=(2, 3, 4))
        volume = volume[:, :, :D].abs()
        return volume[0]
