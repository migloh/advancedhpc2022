import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit

img = plt.imread('images/bird.png')
lumi = float(sys.argv[1])/255

img_h, img_w, _ = img.shape


devOutput = cuda.device_array((img_h, img_w, 3), img.dtype)

devInput = cuda.to_device(img)


@jit
def mapfunc(val, thre):
    a = val+thre
    if a > 1:
        a = 1
    elif a < 0:
        a = 0
    return a


@cuda.jit
def brightnessctl(src, dst, thre):
    tr = np.float32(thre)
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    dst[tidx, tidy, 0] = np.float32(mapfunc(src[tidx, tidy, 0], tr))
    dst[tidx, tidy, 1] = np.float32(mapfunc(src[tidx, tidy, 1], tr))
    dst[tidx, tidy, 2] = np.float32(mapfunc(src[tidx, tidy, 2], tr))


blSz = 32
gridSize = (math.ceil(img_h/blSz), math.ceil(img_w/blSz))
blockSize = (blSz, blSz)

start_cuda = time.time()
brightnessctl[gridSize, blockSize](devInput, devOutput, lumi)
stop_cuda = time.time()
print('Cuda: ', stop_cuda-start_cuda)
hostOutput = devOutput.copy_to_host()
plt.imsave('images/brightnessctl.png', hostOutput)
