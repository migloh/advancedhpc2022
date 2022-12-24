import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image

img = plt.imread('images/bird.png')
img_h = img.shape[0]
img_w = img.shape[1]
rgb_to_gray = [0.2989, 0.5870, 0.1140]

grayimg = np.array(img, copy=True)

start_man = time.time()
for i in range(img_h):
    for j in range(img_w):
        px = img[i][j]
        avgPx = (px[0]+px[1]+px[2])/3
        grayimg[i][j][0] = grayimg[i][j][1] = grayimg[i][j][2] = avgPx
stop_man = time.time()

plt.imsave('images/cpu2d.png', grayimg)

# cpu2d = Image.fromarray(grayimg)
# cpu2d.save('images/cpu2d.png')

print("Manual transformation time: ", stop_man-start_man)

grayimg = np.array(img, copy=True)
devOutput = cuda.device_array(img.shape, img.dtype)

devInput = cuda.to_device(img)


@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    g = (src[tidx, tidy, 0] + src[tidx, tidy, 1] + src[tidx, tidy, 2])/3
    dst[tidx, tidy, 0] = dst[tidx, tidy, 1] = dst[tidx, tidy, 2] = g


blSz = 32
gridSize = (np.uint(img_h/blSz), np.uint(img_w/blSz))
blockSize = (blSz, blSz)

start_cuda = time.time()
grayscale[gridSize, blockSize](devInput, devOutput)
stop_cuda = time.time()

hostOutput = devOutput.copy_to_host()
plt.imsave('images/gpu2d.png', hostOutput)
print('Cuda transformation time: ', stop_cuda-start_cuda)

block_size = [2, 4, 8, 16, 32]
collapsed_time = []

for bs in block_size:
    duration = []
    for t in range(3):
        gs = (np.uint(img_h/bs), np.uint(img_w/bs))
        bls = (bs, bs)
        sta = time.time()
        grayscale[gs, bls](devInput, devOutput)
        sto = time.time()
        duration.append(sto-sta)
    collapsed_time.append(sum(duration)/len(duration))

fig, ax = plt.subplots()
ax.plot(block_size, collapsed_time)
ax.set_xlabel('Block size')
ax.set_ylabel('Execution time')
plt.show()
