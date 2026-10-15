#include "common.cuh"
#include <iostream>
#include <vector>

__device__ float warpReduce(float val){
    for(int offset=16;offset>0;offset/=2){
        val+= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
__global__ void reductionSumWarp(const float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    __shared__ float warp_sums[8];

    // 1. Load from global memory
    float my_val = (i < N) ? input[i] : 0.0f;

    // 2. Reduce within each warp
    my_val = warpReduce(my_val);

    // 3. First thread of each warp saves to shared memory
    if (laneId == 0) {
        warp_sums[warpId] = my_val;
    }
    
    // 4. Ensure all 8 warps have finished writing
    __syncthreads();

    // 5 & 6. Only warp 0 finishes the job
    if (warpId == 0) {
        my_val = (laneId < 8) ? warp_sums[laneId] : 0.0f;
        my_val = warpReduce(my_val);

        if (laneId == 0) {
            atomicAdd(output, my_val);
        }
    }
}
__global__ void reductionSumShared(const float * input, float* output,  int N){
    int i =blockIdx.x*blockDim.x+threadIdx.x;
    extern __shared__ float sdata[];
    int j=threadIdx.x;
    if (i < N) {
        sdata[j] = input[i];
    } else {
        sdata[j] = 0.0f;
    }
    __syncthreads();
    
    
    for(int step=1;step<blockDim.x;step*=2){
        if(j%(step*2)==0){
            if(j+step<blockDim.x){
                sdata[j]+=sdata[j+step];
            }
        }
        
        __syncthreads();
    }

   
    if(j==0)
    output[blockIdx.x]=sdata[0];
    
    
}
__global__ void reductionSumSequential(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();
    
// Start with a stride spanning exactly half the block, and halve it each step
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        
        // The active threads are simply the first 'stride' threads
        if (tid < stride) {
            // Because we use 'tid', adjacent threads access adjacent memory banks!
            sdata[tid] += sdata[tid + stride];
        }
        
        // Synchronize threads before the next reduction step
        __syncthreads(); 
    }
    
    if (tid == 0) {
        // For benchmarking speed, writing partial sums is fine.
        // For a final answer, use atomicAdd(output, sdata[0]);
        output[blockIdx.x] = sdata[0]; 
    }
}

int main() {
    // Test with 100 million elements (approx 400 MB of data)
    int N = 100000000;
    size_t bytes = N * sizeof(float);
    
    // Allocate Host Memory
    std::vector<float> h_input(N, 1.0f); // Fill with 1.0f for easy verification
    
    // Allocate Device Memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    
    // The naive/sequential kernels output partial sums per block, 
    // so the output array needs to be sized to the grid.
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMalloc(&d_output, blocks * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    
    printf("Benchmarking Reduction (N = %d elements, %.1f MB)\n", N, bytes / 1e6);
    printf("------------------------------------------------------\n");

    // 1. Benchmark: Interleaved Shared Memory (High Warp Divergence)
    BENCH("Shared Diverged", bytes, 
          reductionSumShared<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, N)
    );

    // 2. Benchmark: Sequential Shared Memory (Zero Warp Divergence)
    BENCH("Shared Sequent", bytes, 
          reductionSumSequential<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, N)
    );

    // 3. Benchmark: Warp Shuffle (Registers Only, No Shared Memory bottleneck)
    // Note: If warp shuffle uses atomicAdd inside, output size is just 1 float.
    BENCH("Warp Shuffle", bytes, 
          reductionSumWarp<<<blocks, threads>>>(d_input, d_output, N)
    );

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}