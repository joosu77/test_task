import glob
import cv2
from numba import cuda
import numpy as np
import time

# Pixel counts that fit criteria for each picture
check_arr = [1512,1524,1520,1524,1517,1520,1526,1531,1522,1511,1529,1520,1536,1532]

@cuda.jit('void(uint8[:,:], uint8[:,:], uint8, uint16, uint16 , int32[:], int32[:,:])')
def cuda_kernel(input, output, threshold, W, H, counter, coords):
    startX, startY = cuda.grid(2) 
    stepX = cuda.gridDim.x * cuda.blockDim.x 
    stepY = cuda.gridDim.y * cuda.blockDim.y
    for x in range(startX, W, stepX):
        for y in range(startY, H, stepY):
            # if pixel satisfies criteria make it whyte
            if input[y,x] + threshold < input[(y+50)%H,(x+50)%W]:
                output[y,x] = 255
                # Store whyte pixel coordinates and counts
                idx = cuda.atomic.add(counter, 0, 1)
                coords[idx, 0] = x
                coords[idx, 1] = y
            else:
                output[y,x] = 0

def main():
    # Grayscale and resize images
    frames = sorted(glob.glob("media/*.png"))
    frames = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY) for x in frames]
    frames = [cv2.resize(x, (256,256)) for x in frames]
    
    # Time variables
    cum_times = 0
    num_times = 0
    
    # blockdim and griddim define how many cuda workers are used
    blockdim = (16,16)
    griddim = (32,32)
    
    # Pre-allocate arrays for pixel coordinates (AI said so)
    d_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_coords = cuda.device_array((256*256, 2), dtype=np.int32) # Max coords
    while 1:
        for ctr in range(len(frames)):
            start_time = time.time()
            H,W = frames[ctr].shape
            
            cuda.to_device(np.zeros(1, dtype=np.int32), to=d_counter)
            d_image = cuda.to_device(frames[ctr])
            d_output = cuda.device_array(frames[ctr].shape, dtype=frames[ctr].dtype)
            cuda_kernel[blockdim,griddim](d_image, d_output, 150, W, H, d_counter, d_coords)
            
            # Results counts and coordinates
            output = d_counter.copy_to_host()[0]
            coords = d_coords.copy_to_host()[:output]
            found_pixels = [(x, y) for x, y in coords]

            # This list should not change after optimizations:
            assert len(found_pixels) == check_arr[ctr]
            
            time_taken = time.time()-start_time
            cum_times += time_taken
            num_times += 1
            print(f"Time taken for frame {ctr}: {round(time_taken*1000,2)}ms, average: {round(cum_times/num_times*1000,2)}ms")
        
main()