import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

img = plt.imread('assets/bird.png')
img_h = img.shape[0]
img_w = img.shape[1]
rgb_to_gray = [0.2989, 0.5870, 0.1140]

reshaped_img = np.reshape(img, (img_w*img_h, 3))
grayimg = np.zeros(img_w*img_h)

start_man = time.time()
for i in range(len(reshaped_img)):
    grayimg[i] = reshaped_img[i][0]*rgb_to_gray[0] + reshaped_img[i][1] * \
        rgb_to_gray[1] + reshaped_img[i][2]*rgb_to_gray[2]
stop_man = time.time()

print("Manual transformation time: ", stop_man-start_man)

grayimg = np.reshape(grayimg, (img_h, img_w))

devOutput = cuda.device_array((img_h*img_w), np.float64)
devInput = cuda.to_device(reshaped_img)

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = (src[tidx, 0] + src[tidx, 1] + src[tidx, 2])/3
    dst[tidx] = g

blockSize = 64 
gridSize = math.ceil((img_h*img_w) / blockSize)

start_cuda = time.time()
grayscale[gridSize, blockSize](devInput, devOutput)
stop_cuda = time.time()

hostOutput = devOutput.copy_to_host()
hostOutput = np.reshape(hostOutput, (img_h, img_w))
print('Cuda transformation time: ', stop_cuda-start_cuda)

block_size = [16, 32, 64, 128, 256, 512, 1024]
collapsed_time = []

for bs in block_size:
    duration = []
    for t in range(3):
        gs = math.ceil((img_h*img_w) / bs)
        sta = time.time()
        grayscale[gs, bs](devInput, devOutput)
        sto = time.time()
        duration.append(sto-sta)
    collapsed_time.append(sum(duration)/len(duration)) 

fig, ax = plt.subplots()
# ax.imshow(hostOutput, cmap='gray', vmin=0, vmax=1)
ax.plot(block_size, collapsed_time)
ax.set_xlabel('Block size')
ax.set_ylabel('Execution time')
plt.show()
