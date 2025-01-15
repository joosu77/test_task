import glob
import cv2
from numba import cuda
import time
import cv2

check_arr = [0,42,42,42,40,43,39,46,41,43,39,42,41,41]

@cuda.jit('void(uint8[:,:], uint8[:,:], uint8, uint8, uint8)')
def cuda_kernel(input, output, threshold, W, H):
    startX, startY = cuda.grid(2)
    stepX = cuda.gridDim.x * cuda.blockDim.x
    stepY = cuda.gridDim.y * cuda.blockDim.y
    for x in range(startX, W, stepX):
        for y in range(startY, H, stepY):
            if input[y,x] + threshold < input[(y+50)%H,(x+50)%W]:
                output[y,x] = 255
            else:
                output[y,x] = 0

def main():
    frames = sorted(glob.glob("media/*.png"))
    frames = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY) for x in frames]
    frames = [cv2.resize(x, (256,256)) for x in frames]
    cum_times = 0
    num_times = 0
    while 1:
        for ctr in range(len(frames)):
            start_time = time.time()
            H,W = frames[ctr].shape
            d_image = cuda.to_device(frames[ctr])
            d_output = cuda.device_array(frames[ctr].shape, dtype=frames[ctr].dtype)
            # blockdim and griddim define how many cuda workers are used
            blockdim = (16,16)
            griddim = (32,32)
            print("start")
            cuda_kernel[blockdim,griddim](d_image, d_output, 50, W, H)
            print("end")
            output = d_output.copy_to_host()
            
            cv2.imshow("1", output)
            cv2.waitKey(1)
            
            found_pixels = []
            for x in range(W):
                for y in range(H):
                    if output[y,x] == 255:
                        found_pixels.append((x,y))
                        
            # This list should not change after optimizations:
            try:
                assert len(found_pixels) == check_arr[ctr]
            except:
                print(ctr, len(found_pixels), check_arr[ctr])
            #print(f"Pixels found for frame {ctr}: {len(found_pixels)}")
            time_taken = time.time()-start_time
            cum_times += time_taken
            num_times += 1
            print(f"Time taken: {time_taken}, average: {round(cum_times/num_times,2)}")
        
main()