"""Video Writer Helper"""
import datetime
from multiprocessing.sharedctypes import Value
import cv2
import torch
import numpy as np
from os import path, makedirs
from typing import Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter

from totri.util.format import tensor_to_chw

class VideoWriter():
    """Wrapper around cv2.VideoWriter

    After writing video files, the OpenCV handles must be released.
    This will be done automatically upon destruction, alternatively
    use VideoSummaryWriter as context manager.
    """

    def __init__(self, file_name: str, resolution: Tuple[int],
                 fps: Optional[int] = None,
                 fourcc: Optional[str] = None):
        """Init VideoWriter

        Args:
            file_name (str): Filename including appropriate extension
            resolution (Tuple[int]): (height, width) tuple
            fps (int, optional): Frames per second. Defaults to 25.
            fourcc (Optional[str], optional): `cv2.VideoWriter_fourcc` code, e.g. `'mp4v'` (default)
        """
        self.file_name = path.abspath(file_name)
        self.resolution = resolution
        self.fps = fps or 25
        self.fourcc = fourcc or cv2.VideoWriter_fourcc(*'mp4v')
        self.dirname = path.dirname(self.file_name)
        if not path.isdir(self.dirname):
            makedirs(self.dirname)
        self._writer = cv2.VideoWriter(
            self.file_name, self.fourcc, self.fps,
            (self.resolution[1], self.resolution[0]))
        self.frame_count = 0
        
    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def write_frame(self, img_tensor: Union[torch.Tensor, np.ndarray],
                    dataformats: str = 'CHW'):
        if self._writer is None:
            raise RuntimeError('Video writer has been closed. Cannot write new frames.')
        with torch.no_grad():
            # Transform to CHW format
            img_tensor = tensor_to_chw(img_tensor, dataformats)
            # Convert to uint8
            if img_tensor.dtype != torch.uint8:
                img_tensor = (img_tensor * 255).to(dtype=torch.uint8)
            # CHW to HWC
            img_tensor = img_tensor.permute(1, 2, 0)
            # Check dimensions
            if (img_tensor.shape[0] != self.resolution[0] or
                img_tensor.shape[1] != self.resolution[1]):
                raise ValueError(
                    f"Tensor has wrong dimensions."
                    f"Expected {self.resolution[0]}x{self.resolution[1]}, "
                    f"but got {img_tensor.shape[0]}x{img_tensor.shape[1]}.")
            # RGB TO BGR
            img_tensor_bgr = img_tensor[..., (2, 1, 0)]
            self._writer.write(img_tensor_bgr.contiguous().cpu().numpy())
            self.frame_count += 1

class RealTimeVideoWriter(VideoWriter):

    def __init__(self, *args, speed=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_img_tensor = None
            self.last_time = None
            self.speed = float(speed)

    def write_frame(self, data: Union[torch.Tensor, np.ndarray],
                    dataformats: str = 'CHW', transform = lambda x: x):
        with torch.no_grad():
            frame_length = datetime.timedelta(seconds=self.speed/self.fps)
            now = datetime.datetime.now()
            if self.last_time is None:
                img_tensor = tensor_to_chw(transform(data), dataformats)
                super().write_frame(img_tensor, 'CHW')
                self.last_img_tensor = img_tensor
                self.last_time = now + frame_length
            elif self.last_time <= now:
                while self.last_time + frame_length < now:
                    super().write_frame(self.last_img_tensor, 'CHW')
                    self.last_time += frame_length
                img_tensor = tensor_to_chw(transform(data), dataformats)
                super().write_frame(img_tensor, 'CHW')
                self.last_img_tensor = img_tensor
                self.last_time += frame_length

class VideoSummaryWriter(SummaryWriter):
    """Extend SummaryWriter to write images as video

    After writing video files, the OpenCV handles must be released.
    This will be done automatically upon destruction, alternatively
    use VideoSummaryWriter as context manager.

    Video files will be put into the corresponding log_dir according
    to the given tag. To write video frames pass `video=True` to
    add_image or add_images.
    """

    def __init__(self, *args, extension: Optional[str] = None,
                 fourcc: Optional[str] = None, fps: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._writers = {}
        self.extension = extension
        self.fourcc = fourcc
        self.fps = fps

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def release(self):
        """Release all writers"""
        for writer in self._writers.values():
            writer.release()

    def _add_video_frame(self, tag: str, img_tensor_chw: torch.Tensor):
        """Add img_tensor to the video corresponding to tag"""
        if tag not in self._writers:
            file_name = path.join(self.log_dir, tag + self.extension)
            resolution = tuple(img_tensor_chw.shape[1:])
            dirname = path.dirname(file_name)
            if not path.isdir(dirname):
                makedirs(dirname)
            self._writers[tag] = VideoWriter(
                file_name, resolution, self.fps, self.fourcc)
        self._writers[tag].write(img_tensor_chw, 'CHW')

    def add_image(self, tag, img_tensor, global_step=None, walltime=None,
                  dataformats='CHW', video=False):
        img_tensor = tensor_to_chw(img_tensor, dataformats)
        super().add_image(tag, img_tensor, global_step, walltime, 'CHW')
        if video:
            self._add_video_frame(tag, img_tensor)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None,
                   dataformats='NCHW', video=False):
        img_tensor = tensor_to_chw(img_tensor, dataformats)
        super().add_image(tag, img_tensor, global_step, walltime, 'CHW')
        if video:
            self._add_video_frame(tag, img_tensor)
