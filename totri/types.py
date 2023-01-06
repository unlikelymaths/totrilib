import typing
import torch

C0 = 299792458.0

Vec3 = typing.Tuple[float, float, float]
Size3 = typing.Tuple[int, int, int]
Vec3_ = typing.Union[Vec3, Size3, int, float]
Size3_ = typing.Union[Size3, int]

def to_vec3(input: Vec3_) -> Vec3:
    if isinstance(input, int) or isinstance(input, float):
        return (input, input, input)
    elif isinstance(input, tuple) or isinstance(input, list):
        if len(input) != 3:
            raise ValueError(
                f'Vec3 must be tuple of length 3, got {len(input)}')
        for i in range(3):
            if not (isinstance(input[i], int) or isinstance(input[i], float)):
                raise ValueError(
                    f'Vec3 entries must be int or float, got {type(input[i])}')
        return (float(input[0]), float(input[1]), float(input[2]))
    raise ValueError(
        f'Vec3 must be float or tuple, got {type(input)}')

def to_size3(input: Size3_) -> Size3:
    if isinstance(input, int):
        return (input, input, input)
    elif isinstance(input, tuple) or isinstance(input, list):
        if len(input) != 3:
            raise ValueError(
                f'Size3 must be tuple of length 3, got {len(input)}')
        for i in range(3):
            if not (isinstance(input[i], int)):
                raise ValueError(
                    f'Size3 entries must be int, got {type(input[i])}')
        return (input[0], input[1], input[2])
    raise ValueError(
        f'Size3 must be int or tuple, got {type(input)}')

class VolumeDef():
    __slots__ = ['start', 'end', 'resolution']

    def __init__(self, start: Vec3_, end: Vec3_, resolution: Size3_):
        self.start: Vec3 = to_vec3(start)
        self.end: Vec3 = to_vec3(end)
        self.resolution: Size3 = to_size3(resolution)
        """Volume resolution D H W"""

    def __str__(self):
        start_str = ', '.join([f'{value:.3e} m' for value in self.start])
        end_str   = ', '.join([f'{value:.3e} m' for value in self.end])
        res_str   = ', '.join([f'{value}'       for value in self.resolution])
        return f'VolumeDef(start={start_str}, ' \
            f'end={end_str}, resolution={res_str})'

    def clone(self):
        return VolumeDef(self.start, self.end, self.resolution)

    def extent(self):
        """(x, y, z) extent"""
        return tuple(abs(self.start[i] - self.end[i]) for i in range(3))

    def corner_points(self) -> torch.Tensor:
        """Make tensor with corner coordinates

        Returns:
            torch.Tensor: Corner coordinates of the volume [3, 8]
        """
        return torch.Tensor([
            [self.start[0], self.start[1], self.start[2]],
            [self.start[0],   self.end[1], self.start[2]],
            [self.start[0], self.start[1],   self.end[2]],
            [self.start[0],   self.end[1],   self.end[2]],
            [  self.end[0], self.start[1], self.start[2]],
            [  self.end[0],   self.end[1], self.start[2]],
            [  self.end[0], self.start[1],   self.end[2]],
            [  self.end[0],   self.end[1],   self.end[2]],
        ]).T

    def flat_idx_to_3d_idx(self, idx):
        return (
            idx % self.resolution[2],
            (idx // self.resolution[2]) % self.resolution[1],
            (idx // self.resolution[2]) // self.resolution[1],
        )

    def index_to_coordinates(self, idx: Size3_):
        idx = to_size3(idx)
        return (
            idx[0] / self.resolution[2] * (self.end[0] - self.start[0]) + self.start[0],
            idx[1] / self.resolution[1] * (self.end[1] - self.start[1]) + self.start[1],
            idx[2] / self.resolution[0] * (self.end[2] - self.start[2]) + self.start[2],
        )

    def coordinates_inclusive(self, resolution: Size3_ = None,
                              device: torch.device = torch.device('cpu')
                              ) -> torch.Tensor:
        """Make volume of coordinates where the first coordinates are at start
        and the last coordinates are at end.

        Returns:
            torch.Tensor: Corner coordinates of the volume [D, H, W, 3]
                          Entries are (x,y,z) coordinates.
        """
        if resolution is None:
            if self.resolution is None:
                raise ValueError('No resolution specified.')
            resolution = self.resolution
        else:
            resolution = to_size3(resolution)
        x = torch.linspace(self.start[0], self.end[0], resolution[2], device=device)
        y = torch.linspace(self.start[1], self.end[1], resolution[1], device=device)
        z = torch.linspace(self.start[2], self.end[2], resolution[0], device=device)
        coordinates = torch.stack(torch.meshgrid(z, y, x)[::-1], dim=-1)
        return coordinates

class BinDef():
    __slots__ = ['width', 'offset', 'num']

    def __init__(self, width: float, offset: float=0, num: typing.Optional[int] = None,
                 unit_is_time: bool=False):
        """Makes a bin definition

        Args:
            width (float): bin width in either meter or seconds
            offset (float): offset of the first bin in either meter or seconds
            num (int, optional): Number of bins. Defaults to None
            unit_is_time (bool, optional): If true, with and offset are given in seconds.
                Otherwise they are given in meter and converted to seconds. Defaults to True.
        """
        if unit_is_time:
            width  *= C0
            offset *= C0
        self.width = width
        self.offset = offset
        self.num = num

    def __str__(self):
        return f'BinDef(width={self.width/C0:.3e}s [{self.width:.3e}m], ' \
            f'offset={self.offset/C0:.3e}s [{self.offset:.3e}m], num={self.num})'

    @property
    def width_s(self):
        return self.width / C0

    @width_s.setter
    def width_s(self, value):
        self.width = value * C0

    @property
    def offset_s(self):
        return self.offset / C0

    @offset_s.setter
    def offset_s(self, value):
        self.offset = value * C0
