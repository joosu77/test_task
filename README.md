# Task description
This code naively implements finding all pixels that correspond to the criteria that the pixel `(x,y)` has a lower value than the pixel at location `(x+50,y+50)` by at least some
# given threshold.
This implementation uses a cuda kernel that iterates over the input image, marks all pixels that match the criteria on the output image, and then iterates over the output image on the CPU to gather all the marked pixels into a single list.
# Objective
Optimize the code such that it finds the same lists of pixels as the given code but does it faster. Note that there are only automated assertions for checking that the number of pixels matches but you should still find all the same pixels as the original code not just the number of pixels.
## Hint
Iterating over the whole image on a CPU is time consuming, so iterating over the whole output image is the main bottleneck.

For reference, on my system time taken per image is about 150ms on average, but after some optimisations it takes <15ms.