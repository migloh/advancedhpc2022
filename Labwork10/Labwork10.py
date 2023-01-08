import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import cv2

img = plt.imread('images/bird.png')
wsize = int(sys.argv[1])

padded_img = cv2.copyMakeBorder(
    img, wsize, wsize, wsize, wsize, cv2.BORDER_CONSTANT, None, value=0)
img_h, img_w, _ = padded_img.shape

hsvOutput = cuda.device_array((3, img_h, img_w), img.dtype)
rgbFinal = cuda.device_array(padded_img.shape, img.dtype)

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


def KuwaharaCPU(srcRGB, srcHSV, padsize):
    finalrgb = np.zeros(shape=srcRGB.shape, dtype=np.float32)

    for tidx in range(srcRGB.shape[0]):
        for tidy in range(srcRGB.shape[1]):
            coorList = (((tidx-padsize, tidx+1), (tidy-padsize, tidy+1)),
                        ((tidx, tidx+padsize+1), (tidy-padsize, tidy+1)),
                        ((tidx-padsize, tidx+1), (tidy, tidy+padsize+1)),
                        ((tidx, tidx+padsize+1), (tidy, tidy+padsize+1)))
            if tidx < padsize or tidx >= srcRGB.shape[0] - padsize or tidy < padsize or tidy >= srcRGB.shape[1] - padsize:
                finalrgb[tidx, tidy, 0] = finalrgb[tidx,
                                                   tidy, 1] = finalrgb[tidx, tidy, 2] = 0
            else:
                avg_r = np.float32(0.0)
                avg_g = np.float32(0.0)
                avg_b = np.float32(0.0)

                minid = 0
                minstd = 9999.0

                for co in range(4):
                    sam = np.float32(0.0)
                    tempsqrd = np.float32(0.0)

                    for i in range(*coorList[co][0]):
                        for j in range(*coorList[co][1]):
                            sam += srcHSV[2, i, j]
                            tempsqrd += srcHSV[2, i, j]**2

                    n = (padsize+1)**2
                    tempMean = sam/n
                    stdd = math.sqrt(abs((tempsqrd / n) - tempMean**2))
                    if stdd < minstd:
                        minstd = stdd
                        minid = co

                coormin = coorList[minid]

                for i in range(*coormin[0]):
                    for j in range(*coormin[1]):
                        avg_r += srcRGB[i, j, 0]
                        avg_g += srcRGB[i, j, 1]
                        avg_b += srcRGB[i, j, 2]
                finalrgb[tidx, tidy, 0] = avg_r / (padsize+1)**2
                finalrgb[tidx, tidy, 1] = avg_g / (padsize+1)**2
                finalrgb[tidx, tidy, 2] = avg_b / (padsize+1)**2
    return finalrgb


@ cuda.jit
def KuwaharaGPU(srcRGB, srcHSV, dst, padsize):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if tidx < padsize or tidx >= srcRGB.shape[0] - padsize or tidy < padsize or tidy >= srcRGB.shape[1] - padsize:
        dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = 0
        return

    avg_r = np.float32(0.0)
    avg_g = np.float32(0.0)
    avg_b = np.float32(0.0)

    minid = 0
    minstd = 9999.0
    coorList = (((tidx-padsize, tidx+1), (tidy-padsize, tidy+1)),
                ((tidx, tidx+padsize+1), (tidy-padsize, tidy+1)),
                ((tidx-padsize, tidx+1), (tidy, tidy+padsize+1)),
                ((tidx, tidx+padsize+1), (tidy, tidy+padsize+1)))

    for co in range(4):
        sam = np.float32(0.0)
        tempsqrd = np.float32(0.0)

        for i in range(*coorList[co][0]):
            for j in range(*coorList[co][1]):
                sam = sam + srcHSV[2, i, j]
                tempsqrd = tempsqrd + srcHSV[2, i, j]**2

        n = (padsize+1)**2
        tempMean = sam/n
        stdd = math.sqrt(abs(tempsqrd / n - tempMean**2))
        if stdd < minstd:
            minstd = stdd
            minid = co

    coormin = coorList[minid]

    for i in range(*coormin[0]):
        for j in range(*coormin[1]):
            avg_r += srcRGB[i, j, 0]
            avg_g += srcRGB[i, j, 1]
            avg_b += srcRGB[i, j, 2]

    dst[tidx, tidy, 0] = avg_r / (padsize+1)**2
    dst[tidx, tidy, 1] = avg_g / (padsize+1)**2
    dst[tidx, tidy, 2] = avg_b / (padsize+1)**2


rgbInput = cuda.to_device(padded_img)
RGB2HSV[gridSize, blockSize](rgbInput, hsvOutput)

hsvmid = hsvOutput.copy_to_host()
vmid = hsvmid[2]
vinput = cuda.to_device(vmid)

start_cuda = time.time()
KuwaharaGPU[gridSize, blockSize](rgbInput, hsvOutput, rgbFinal, wsize)
stop_cuda = time.time()

rgbimg = rgbFinal.copy_to_host()
hsvout = hsvOutput.copy_to_host()
plt.imsave('images/gputransformed.png', rgbimg)
print('Kuwahara using GPU: ', stop_cuda-start_cuda)

start_cuda = time.time()
imgcpu = KuwaharaCPU(padded_img, hsvout, wsize)
stop_cuda = time.time()
print('Kuwahara using CPU: ', stop_cuda-start_cuda)
rgbimg = rgbFinal.copy_to_host()
plt.imsave('images/cputransformed.png', rgbimg)
