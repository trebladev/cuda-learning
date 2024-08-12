#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cfloat> 

#define WARP_SIZE 32

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(max(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

template <typename T>
__device__ __forceinline__ T WarpReduceMax(T val){
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset >=1; offset /= 2){
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template <typename T>
__global__ void ReduceMax(T* src, T* dst, int n){
    extern __shared__ T shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    T val = (bid*blockDim.x+tid < n) ? src[bid*blockDim.x+tid] : T(-FLT_MAX);
    val = WarpReduceMax(val);

    if (lane_id == 0){
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0){
        val = (lane_id < blockDim.x / WARP_SIZE) ? shared[lane_id] : T(-FLT_MAX);
        val = WarpReduceMax(val);
    }

    if (tid == 0){
        atomicMaxFloat(dst, val);
    }
}

int main() {
    const int n = 10240;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> h_data(n, 1.0f); // 初始化一个大小为 n 的向量，值为 1.0
    h_data[500] = 10.0f; // 在某个位置放一个最大值

    float* d_data = nullptr;
    float* d_result = nullptr;
    float h_result = -FLT_MAX;

    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    ReduceMax<<<blocksPerGrid, threadsPerBlock, threadsPerBlock / WARP_SIZE * sizeof(float)>>>(d_data, d_result, n);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Maximum value: " << h_result << std::endl;

    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}