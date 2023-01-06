import copy
import torch
import numpy as np

class TransientBackgroundBase(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def clone(self):
        return copy.deepcopy(self)

    def param_list(self):
        return list(self.parameters())

    def clamp_parameters(self, foreground, background):
        pass

    def requires_grad(self, requires_grad: bool):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def copy_from(self, other):
        for param_this, param_other in zip(self.parameters(), other.parameters()):
            param_this[:] = param_other[:]

class TransientBackgroundEmpty(TransientBackgroundBase):

    def __init__(self):
        super().__init__()

    def forward(self, *args) -> int:
        if len(args) == 1:
            scan_points = args[0]
            return torch.zeros(
                (scan_points.shape[0], 1, scan_points.shape[1]),
                device=scan_points.device, dtype=scan_points.dtype)
        elif len(args) == 2:
            scan_points = args[0]
            laser_points = args[1]
            return torch.zeros(
                (scan_points.shape[0], 1, scan_points.shape[1], laser_points.shape[1]),
                device=scan_points.device, dtype=scan_points.dtype)
        else:
            raise ValueError('Unexpected number of arguments')

class TransientBackgroundConfocal(TransientBackgroundBase):

    def __init__(self, num_bins, wall_start=(-1,-1), wall_end=(1,1), num_pos_encodings=2, num_t_encodings=4, channels=8, clamping_factor = 1.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.num_bins = num_bins
        self.num_bins_sub = max(num_bins // 8, 1)
        self.shift = (
            -min(wall_start[0], wall_end[0]),
            -min(wall_start[1],  wall_end[1])
        )
        self.normalization = (
            np.pi / abs(wall_start[0] - wall_end[0]),
            np.pi / abs(wall_start[1] - wall_end[1])
        )
        self.num_pos_encodings = num_pos_encodings
        self.num_t_encodings = num_t_encodings
        self.clamping_factor = clamping_factor
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2*num_pos_encodings + num_t_encodings, channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels, channels//2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels//2, 1),
            torch.nn.Sigmoid(),
        ) # [B, T0, S, 2*PE+TE] -> [B, T0, S, 1]
        self.intensity = torch.nn.parameter.Parameter(torch.ones((1,)))
        self.to(device=device, dtype=dtype)

    def forward(self, scan_points: torch.Tensor) -> torch.Tensor:
        """Predict transient background for each scan point

        Args:
            scan_points (torch.Tensor): [B, S, 3]

        Returns:
            torch.Tensor: transient[B, T, S]
        """
        # Spatial Encoding
        x = (scan_points[:, :, 0:1] + self.shift[0]) * self.normalization[0] # [B, S, 1] in range[0, pi]
        y = (scan_points[:, :, 1:2] + self.shift[1]) * self.normalization[1] # [B, S, 1] in range[0, pi]
        pos_encoding = torch.cat(
            [torch.cos((i + 1) * x) for i in range(self.num_pos_encodings)] +
            [torch.cos((i + 1) * y) for i in range(self.num_pos_encodings)],
            dim=-1)[:, None, :, :].expand(-1, self.num_bins_sub, -1, -1) # [B, T0, S, 2*PE]
        # Temporal encoding
        t = torch.linspace(0, np.pi, self.num_bins_sub, dtype=scan_points.dtype, device=scan_points.device).view(1, self.num_bins_sub, 1, 1) # [1, T0, 1, 1]
        t = t.expand(scan_points.shape[0], -1, scan_points.shape[1], 1) # [B, T0, S, 1]
        t_encoding = torch.cat(
            [torch.cos((i + 1) * t) for i in range(self.num_t_encodings)],
            dim=-1) # [B, T0, S, TE]
        # Network
        background = self.decoder(torch.cat((pos_encoding, t_encoding), dim=-1))[:,:,:,0] # [B, T0, S]
        background = torch.nn.functional.interpolate(
            background.permute(0, 2, 1),
            self.num_bins, mode='linear', align_corners=False
            ).permute(0, 2, 1)  # [B, T, S]
        return self.intensity * background

    def clamp_parameters(self, foreground, background):
        with torch.no_grad():
            foreground_energy = (foreground**2).mean()**0.5
            background_energy = (background**2).mean()**0.5
            max_energy = self.clamping_factor * foreground_energy
            if background_energy > max_energy:
                self.intensity[:] *= max_energy / background_energy

class TransientBackgroundExhaustive(TransientBackgroundBase):

    def __init__(self, num_bins, wall_start=(-1,-1), wall_end=(1,1), num_pos_encodings=2, num_t_encodings=4, channels=8, device='cpu', dtype=torch.float32):
        super().__init__()
        self.num_bins = num_bins
        self.num_bins_sub = max(num_bins // 8, 1)
        self.shift = (
            -min(wall_start[0], wall_end[0]),
            -min(wall_start[1],  wall_end[1])
        )
        self.normalization = (
            np.pi / abs(wall_start[0] - wall_end[0]),
            np.pi / abs(wall_start[1] - wall_end[1])
        )
        self.num_pos_encodings = num_pos_encodings
        self.num_t_encodings = num_t_encodings
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4*num_pos_encodings + num_t_encodings, channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels, channels//2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels//2, 1),
            torch.nn.Sigmoid(),
        ) # [B, T0, S, L, 4*PE+TE] -> [B, T0, S, L, 1]
        self.intensity = torch.nn.parameter.Parameter(torch.ones((1,)))
        self.to(device=device, dtype=dtype)

    def forward(self, scan_points: torch.Tensor, laser_points:torch.Tensor) -> torch.Tensor:
        """Predict transient background for each scan combination

        Args:
            scan_points (torch.Tensor): [B, S, 3]
            laser_points (torch.Tensor): [B, S, 3]

        Returns:
            torch.Tensor: transient[B, T, S, L]
        """
        # Spatial Encoding
        B = scan_points.shape[0]
        S = scan_points.shape[1]
        L = laser_points.shape[1]
        x_scan  = ( scan_points[:, :, 0:1] + self.shift[0]) * self.normalization[0] # [B, S, 1] in range[0, pi]
        y_scan  = ( scan_points[:, :, 1:2] + self.shift[1]) * self.normalization[1] # [B, S, 1] in range[0, pi]
        x_laser = (laser_points[:, :, 0:1] + self.shift[0]) * self.normalization[0] # [B, L, 1] in range[0, pi]
        y_laser = (laser_points[:, :, 1:2] + self.shift[1]) * self.normalization[1] # [B, L, 1] in range[0, pi]
        pos_encoding_scan = torch.cat(
            [torch.cos((i + 1) * x_scan) for i in range(self.num_pos_encodings)] +
            [torch.cos((i + 1) * y_scan) for i in range(self.num_pos_encodings)],
            dim=-1)[:, None, :, None, :].expand(-1, self.num_bins_sub, -1, L, -1) # [B, T0, S, L, 2*PE]
        pos_encoding_laser = torch.cat(
            [torch.cos((i + 1) * x_laser) for i in range(self.num_pos_encodings)] +
            [torch.cos((i + 1) * y_laser) for i in range(self.num_pos_encodings)],
            dim=-1)[:, None, None, :, :].expand(-1, self.num_bins_sub, S, -1, -1) # [B, T0, S, L, 2*PE]
        # Temporal encoding
        t = torch.linspace(0, np.pi, self.num_bins_sub, dtype=scan_points.dtype, device=scan_points.device).view(1, self.num_bins_sub, 1, 1, 1) # [1, T0, 1, 1, 1]
        t = t.expand(scan_points.shape[0], -1, S, L, 1) # [B, T0, S, L, 1]
        t_encoding = torch.cat(
            [torch.cos((i + 1) * t) for i in range(self.num_t_encodings)],
            dim=-1) # [B, T0, S, L, TE]
        # Network
        background = self.decoder(torch.cat((pos_encoding_scan, pos_encoding_laser, t_encoding), dim=-1))[:,:,:,:,0] # [B, T0, S, L]
        background = torch.nn.functional.interpolate(
            background.view(B, self.num_bins_sub, S*L).permute(0, 2, 1),
            self.num_bins, mode='linear', align_corners=False
            ).permute(0, 2, 1).view(B, self.num_bins, S, L)  # [B, T, S, L]
        return self.intensity * background

    def clamp_parameters(self, foreground, background):
        with torch.no_grad():
            foreground_energy = (foreground**2).mean()**0.5
            background_energy = (background**2).mean()**0.5
            max_energy = 1.0 * foreground_energy
            if background_energy > max_energy:
                self.intensity[:] *= max_energy / background_energy
