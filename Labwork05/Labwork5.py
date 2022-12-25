import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image
import cv2

img = plt.imread('images/bird.png')
img_h = img.shape[0]
img_w = img.shape[1]

gauss = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0],
])
gauss_sum = np.sum(gauss)

img_pad = cv2.copyMakeBorder(
    img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, value=0)
print(img.dtype)
print(img_pad.dtype)

blurimg = np.array(img, copy=True)

start_man = time.time()
for i in range(3, img_pad.shape[0]-3):
    for j in range(3, img_pad.shape[1]-3):
        rr = 0
        gg = 0
        bb = 0
        for ii in range(0, 7):
            for jj in range(0, 7):
                rr += img_pad[i-3+ii, j-3+jj, 0]*gauss[ii, jj]
                gg += img_pad[i-3+ii, j-3+jj, 1]*gauss[ii, jj]
                bb += img_pad[i-3+ii, j-3+jj, 2]*gauss[ii, jj]
        blurimg[i-3, j-3, 0] = np.float32(rr/gauss_sum)
        blurimg[i-3, j-3, 1] = np.float32(gg/gauss_sum)
        blurimg[i-3, j-3, 2] = np.float32(bb/gauss_sum)
stop_man = time.time()

plt.imsave('images/cpu2d.png', blurimg)

print("Manual transformation time: ", stop_man-start_man)

# grayimg = np.array(img, copy=True)
# devOutput = cuda.device_array(img.shape, img.dtype)

# devInput = cuda.to_device(img_pad)


# @cuda.jit
# def grayscale(src, dst):
#     tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#     tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
#     gauss = np.array([
#         [0, 0, 1, 2, 1, 0, 0],
#         [0, 3, 13, 22, 13, 3, 0],
#         [1, 13, 59, 97, 59, 13, 1],
#         [2, 22, 97, 159, 97, 22, 2],
#         [1, 13, 59, 97, 59, 13, 1],
#         [0, 3, 13, 22, 13, 3, 0],
#         [0, 0, 1, 2, 1, 0, 0],
#     ])

#     rr = 0
#     gg = 0
#     bb = 0

#     for i in gauss.shape[0]:
#         for j in gauss.shape[1]:
#             if tidx < 3 or tidx > src.shape[0]- 3 or tidy < 3 or tidy > src.shape[1]-3
#             g = (src[tidx, tidy, 0] + src[tidx, tidy, 1] + src[tidx, tidy, 2])/3

#     dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = g


# blSz = 32
# gridSize = (np.uint(img_h/blSz), np.uint(img_w/blSz))
# blockSize = (blSz, blSz)

# start_cuda = time.time()
# grayscale[gridSize, blockSize](devInput, devOutput)
# stop_cuda = time.time()

# hostOutput = devOutput.copy_to_host()
# plt.imsave('images/gpu2d.png', hostOutput)
# print('Cuda transformation time: ', stop_cuda-start_cuda)

# block_size = [2, 4, 8, 16, 32]
# collapsed_time = []

# for bs in block_size:
#     duration = []
#     for t in range(3):
#         gs = (np.uint(img_h/bs), np.uint(img_w/bs))
#         bls = (bs, bs)
#         sta = time.time()
#         grayscale[gs, bls](devInput, devOutput)
#         sto = time.time()
#         duration.append(sto-sta)
#     collapsed_time.append(sum(duration)/len(duration))

# fig, ax = plt.subplots()
# ax.plot(block_size, collapsed_time)
# ax.set_xlabel('Block size')
# ax.set_ylabel('Execution time')
# plt.show()
