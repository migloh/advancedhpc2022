import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

img = plt.imread('images/bird.png')

img_h, img_w, _ = img.shape

hsvOutput = cuda.device_array((3, img_h, img_w), img.dtype)
rgbOutput = cuda.device_array(img.shape, img.dtype)

blockSize = (4, 4)
gridSize = (math.ceil((img_h)/blockSize[0]), math.ceil((img_w)/blockSize[1]))


@cuda.jit
def RGB2HSV(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    mx = max(src[tidx, tidy, 0], src[tidx, tidy, 1], src[tidx, tidy, 2])
    mn = min(src[tidx, tidy, 0], src[tidx, tidy, 1], src[tidx, tidy, 2])
    delta = mx - mn

    if delta == 0:
        dst[0, tidx, tidy] = 0
    elif mx == src[tidx, tidy, 0]:
        dst[0, tidx, tidy] = 60 * \
            (((src[tidx, tidy, 1]-src[tidx, tidy, 2])/delta) % 6)
    elif mx == src[tidx, tidy, 1]:
        dst[0, tidx, tidy] = 60 * \
            (((src[tidx, tidy, 2]-src[tidx, tidy, 0])/delta)+2)
    elif mx == src[tidx, tidy, 2]:
        dst[0, tidx, tidy] = 60 * \
            (((src[tidx, tidy, 0]-src[tidx, tidy, 1])/delta)+4)

    if mx == 0:
        dst[1, tidx, tidy] = 0
    else:
        dst[1, tidx, tidy] = delta/mx

    dst[2, tidx, tidy] = mx


@cuda.jit
def HSV2RGB(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    d = src[0, tidx, tidy]/60
    hi = int(d) % 6
    f = d - hi

    l = src[2, tidx, tidy]*(1-src[1, tidx, tidy])
    m = src[2, tidx, tidy]*(1-f*src[1, tidx, tidy])
    n = src[2, tidx, tidy]*(1-(1-f)*src[1, tidx, tidy])

    if 0 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 60:
        dst[tidx, tidy, 0] = src[2, tidx, tidy]
        dst[tidx, tidy, 1] = n
        dst[tidx, tidy, 2] = l
    elif 60 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 120:
        dst[tidx, tidy, 0] = m
        dst[tidx, tidy, 1] = src[2, tidx, tidy]
        dst[tidx, tidy, 2] = l
    elif 120 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 180:
        dst[tidx, tidy, 0] = l
        dst[tidx, tidy, 1] = src[2, tidx, tidy]
        dst[tidx, tidy, 2] = n
    elif 180 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 240:
        dst[tidx, tidy, 0] = l
        dst[tidx, tidy, 1] = m
        dst[tidx, tidy, 2] = src[2, tidx, tidy]
    elif 240 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 300:
        dst[tidx, tidy, 0] = n
        dst[tidx, tidy, 1] = l
        dst[tidx, tidy, 2] = src[2, tidx, tidy]
    elif 300 <= src[0, tidx, tidy] and src[0, tidx, tidy] < 360:
        dst[tidx, tidy, 0] = src[2, tidx, tidy]
        dst[tidx, tidy, 1] = l
        dst[tidx, tidy, 2] = m


start_cuda = time.time()
devInput = cuda.to_device(img)
RGB2HSV[gridSize, blockSize](devInput, hsvOutput)
stop_cuda = time.time()
print('RGB to HSV: ', "{:.2f}".format(stop_cuda-start_cuda), 's')
hostOutput = hsvOutput.copy_to_host()

dim = cuda.to_device(hostOutput)
start_cuda = time.time()
HSV2RGB[gridSize, blockSize](dim, rgbOutput)
stop_cuda = time.time()
print('HSV to RGB: ', "{:.2f}".format(stop_cuda-start_cuda), 's')
ori = rgbOutput.copy_to_host()
plt.imsave('images/transformed.png', ori)
