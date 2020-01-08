import torch
import sys
from subprocess import call
print('__Python VERSION:  sys.version', sys.version)
print('__pyTorch VERSION: torch.__version__', torch.__version__)
print('__CUDA VERSION call(["nvcc", "--version"])')
call(["nvcc", "--version"])
print('__CUDNN VERSION: torch.backends.cudnn.version()', torch.backends.cudnn.version())
print('__Number CUDA Devices: torch.cuda.device_count()', torch.cuda.device_count())
print('__Devices: from nvidia-smi')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print ('Current cuda device: torch.cuda.current_device() ', torch.cuda.current_device())
