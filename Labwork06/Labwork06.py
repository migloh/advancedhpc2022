import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit

img = plt.imread('images/bnw.png')
threshold = float(sys.argv[1])/255

img_h, img_w, _ = img.shape


devOutput = cuda.device_array((img_h, img_w, 3), img.dtype)

devInput = cuda.to_device(img)


@jit
def mapfunc(val, thre):
    a = 0 if val < thre else 1
    return a

# binarization


@cuda.jit
def thresholding(src, dst, thre):
    tr = np.float32(thre)
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    g = np.float32(mapfunc(src[tidx, tidy, 0], tr))
    dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = g


blSz = 32
gridSize = (math.ceil(img_h/blSz), math.ceil(img_w/blSz))
blockSize = (blSz, blSz)

start_cuda = time.time()
thresholding[gridSize, blockSize](devInput, devOutput, threshold)
stop_cuda = time.time()
print('Cuda: ', stop_cuda-start_cuda)
hostOutput = devOutput.copy_to_host()
plt.imsave('images/threshold.png', hostOutput)
