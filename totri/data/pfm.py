import torch
import numpy as np

def PfmTransient(file_path: str, shape: tuple):
    with open(file_path, 'rb') as pfm_file:
        # Check 'pf'
        if not pfm_file.readline().decode().strip().lower().startswith('pf'):
            raise IOError(f'File "{file_path}" is no valid Pfm file.')
        # Size
        given_shape = (
            int(s)
            for s in pfm_file.readline().decode().strip().split())
        # Check -1
        if not float(pfm_file.readline().decode().strip()) == -1:
            raise IOError(f'Line 3 of "{file_path}" must be -1')
        # Read bytes
        array = np.fromfile(pfm_file, dtype=np.float32)
    # Reshape as tensor
    tensor = torch.tensor(array).view(shape)
    return tensor
