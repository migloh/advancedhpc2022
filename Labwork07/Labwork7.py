import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit

img = plt.imread('images/bnw.png')

img_h, img_w, _ = img.shape


def oneDim(im):
    h, w, d = im.shape
    finimg = np.reshape(im, (h*w, d))
    f = [i[0] for i in finimg]
    return np.array(f)


inn = oneDim(img)

devOutput = cuda.device_array((img_h*img_w, 3), img.dtype)

devInput = cuda.to_device(inn)


@cuda.reduce
def lowPx(lo, hi):
    return min(lo, hi)


@cuda.reduce
def highPx(lo, hi):
    return max(lo, hi)


@jit
def graystretch(g, mi, ma):
    re = (g - mi)/(ma-mi)
    return np.float32(re)


minPx = lowPx(inn)
maxPx = highPx(inn)


@cuda.jit
def gstr(src, dst, mi, ma):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    g = graystretch(src[tidx], mi, ma)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g


blockSize = 64
gridSize = math.ceil((img_h*img_w)/blockSize)

start_cuda = time.time()
gstr[gridSize, blockSize](devInput, devOutput, minPx, maxPx)
stop_cuda = time.time()
print('Cuda: ', stop_cuda-start_cuda)
hostOutput = devOutput.copy_to_host()
hostOutput = np.reshape(hostOutput, (img_h, img_w, 3))
plt.imsave('images/graystretch.png', hostOutput)
