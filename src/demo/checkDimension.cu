#include <utils.h>
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */

__global__ void checkIndex(void) {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
	    "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, 
		blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
	    gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 6;

    // define 1-D grid and 1-D block with 3 threads structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    printf("grid-xyz: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("block-xyz: (%d, %d, %d)\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return(0);
}