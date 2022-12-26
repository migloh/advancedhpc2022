import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit

img = plt.imread('images/bird.png')
img2 = plt.imread('images/lion.png')
coeff = float(sys.argv[1])

img_h, img_w, _ = img.shape

imgb = np.array(img, copy=True)
for h in range(img_h):
    for w in range(img_w):
        imgb[h][w] = img2[h][w]


devOutput = cuda.device_array((img_h, img_w, 3), img.dtype)

devInput = cuda.to_device(img)
devInput2 = cuda.to_device(imgb)


@jit
def mapfunc(val1, val2, c):
    a = c*val1 + (1-c)*val2
    return a


@cuda.jit
def brightnessctl(src1, src2, dst, thre):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    dst[tidx, tidy, 0] = mapfunc(
        src1[tidx, tidy, 0], src2[tidx, tidy, 0], thre)
    dst[tidx, tidy, 1] = mapfunc(
        src1[tidx, tidy, 1], src2[tidx, tidy, 1], thre)
    dst[tidx, tidy, 2] = mapfunc(
        src1[tidx, tidy, 2], src2[tidx, tidy, 2], thre)


blSz = 32
gridSize = (math.ceil(img_h/blSz), math.ceil(img_w/blSz))
blockSize = (blSz, blSz)

start_cuda = time.time()
brightnessctl[gridSize, blockSize](devInput, devInput2, devOutput, coeff)
stop_cuda = time.time()
print('Cuda: ', stop_cuda-start_cuda)
hostOutput = devOutput.copy_to_host()
plt.imsave('images/blending.png', hostOutput)
