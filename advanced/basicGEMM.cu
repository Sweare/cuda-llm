#include <cuda_fp16.h>
#include "common.cuh"
#include <iostream>
#include <vector>
#include <cublas_v2.h>
__global__ void basicGEMM(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta){
    __shared__ half s_a[32][32];
    __shared__ half s_b[32][32];

    //can we fit this on 1 SMs considering 4096 is the maximum K,N,M
    //what does  block contain should we write it memory or only in sahred
    //what is shared emmory we can fit this do we use tiling?
    //we should load alpha and beta into registers?
    //oka what do we load on a thread we load 1 number from the
    //I have 56 SMs so approx 
    float acc = 0.0f;
    for(int k =0 ; k < K ; k += 32){
        int a_global_col=k+threadIdx.x;
        int a_global_row=blockIdx.y*32+threadIdx.y;
        if (a_global_row < M && a_global_col < K) {
            s_a[threadIdx.y][threadIdx.x] = A[a_global_row * K + a_global_col];
        } else {
            s_a[threadIdx.y][threadIdx.x] = __float2half(0.0f); 
        }
        int b_global_row=k+threadIdx.y;
        int b_global_col=blockIdx.x*32+threadIdx.x;
        if (b_global_row < K && b_global_col < N) {
            s_b[threadIdx.y][threadIdx.x] = B[b_global_row * N + b_global_col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = __float2half(0.0f); 
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < 32; ++i){
            acc+=(float)s_a[threadIdx.y][i]*(float)s_b[i][threadIdx.x];
        }
        __syncthreads();
 
    }
    int c_row=blockIdx.y*32+threadIdx.y;
    int c_col = blockIdx.x * 32 + threadIdx.x;
    int c_idx=c_row * N + c_col;
    if (c_row < M && c_col < N) {
        if (beta == 0.0f) {
            C[c_idx] = __float2half(acc * alpha);
        } else {
            C[c_idx] = __float2half(acc * alpha + beta * (float)C[c_idx]);
        }
    }

}
__global__ void basicGEMM_register_tiling_vectorized(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ half s_a[128][32+ 8]; 
    __shared__ half s_b[32][128+ 8];

    float acc[8][8] = {0.0f};
    float reg_a[8];
    float reg_b[8];
    
    // Flatten the 16x16 thread block into a 1D ID from 0 to 255
    int tid = threadIdx.y * blockDim.x + threadIdx.x; 

    for(int k = 0; k < K; k += 32) {
        
 
        // --- Vectorized Load for Matrix A (128 rows x 4 vectors wide) ---
        for (int i = 0; i < 512; i += 256) {
            int vec_idx = i + tid;
            int a_row = vec_idx / 4;               // 4 vectors per row
            int a_col_start = (vec_idx % 4) * 8;   // Actual starting column index
            
            int global_row = blockIdx.y * 128 + a_row;
            int global_col = k + a_col_start;
            
            if (global_row < M && global_col < K) {
                // Cast the ADDRESS, then dereference
                float4 fetched = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
                *reinterpret_cast<float4*>(&s_a[a_row][a_col_start]) = fetched;
            } else {
                // Write a block of zeros (16 bytes = 8 halfs)
                *reinterpret_cast<float4*>(&s_a[a_row][a_col_start]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

   
        // --- Vectorized Load for Matrix B (32 rows x 16 vectors wide) ---
        for (int i = 0; i < 512; i += 256) {
            int vec_idx = i + tid;
            int b_row = vec_idx / 16;              // 16 vectors per row
            int b_col_start = (vec_idx % 16) * 8;
            
            int global_row = k + b_row;
            int global_col = blockIdx.x * 128 + b_col_start;
            
            if (global_row < K && global_col < N) {
                float4 fetched = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
                *reinterpret_cast<float4*>(&s_b[b_row][b_col_start]) = fetched;
            } else {
                *reinterpret_cast<float4*>(&s_b[b_row][b_col_start]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        __syncthreads();
        
       
        #pragma unroll
        for (int i = 0; i < 32; ++i){
            #pragma unroll
            for(int y = 0; y < 8; ++y){
                int a_row = threadIdx.y * 8 + y;
                reg_a[y] = __half2float(s_a[a_row][i]);
            }
            #pragma unroll
            for(int x = 0; x < 8; ++x){
                int b_col = threadIdx.x * 8 + x;
                reg_b[x] = __half2float(s_b[i][b_col]);
            }
            #pragma unroll
            for(int y = 0; y < 8; ++y){
                #pragma unroll
                for(int x = 0; x < 8; ++x) {
                    acc[y][x] += reg_a[y] * reg_b[x];
                }
            }
        }
        __syncthreads();
    }
    
  
    #pragma unroll
    for (int y = 0; y < 8; ++y) {
        #pragma unroll
        for (int x = 0; x < 8; ++x) {
            int c_global_row = blockIdx.y * 128 + threadIdx.y * 8 + y;
            int c_global_col = blockIdx.x * 128 + threadIdx.x * 8 + x;
            
            if (c_global_row < M && c_global_col < N) {
                int c_idx = c_global_row * N + c_global_col;
                if (beta == 0.0f) {
                    C[c_idx] = __float2half(acc[y][x] * alpha);
                } else {
                    C[c_idx] = __float2half(acc[y][x] * alpha + beta * __half2float(C[c_idx]));
                }
            }
        }
    }
}
//memor
__global__ void basicGEMM_register_tiling(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ half s_a[128][32]; 
    __shared__ half s_b[32][128];

    float acc[8][8] = {0.0f};
    float reg_a[8];
    float reg_b[8];
    
    // Flatten the 16x16 thread block into a 1D ID from 0 to 255
    int tid = threadIdx.y * blockDim.x + threadIdx.x; 

    for(int k = 0; k < K; k += 32) {
        
 
        for (int i = 0; i < 4096; i += 256) {
            int idx = i + tid;
            int a_row = idx / 32;
            int a_col = idx % 32;
            
            int global_row = blockIdx.y * 128 + a_row;
            int global_col = k + a_col;
            
            if (global_row < M && global_col < K) {
                s_a[a_row][a_col] = A[global_row * K + global_col];
            } else {
                s_a[a_row][a_col] = __float2half(0.0f); 
            }
        }

   
        for (int i = 0; i < 4096; i += 256) {
            int idx = i + tid;
            int b_row = idx / 128;
            int b_col = idx % 128;
            
            int global_row = k + b_row;
            int global_col = blockIdx.x * 128 + b_col;
            
            if (global_row < K && global_col < N) {
                s_b[b_row][b_col] = B[global_row * N + global_col];
            } else {
                s_b[b_row][b_col] = __float2half(0.0f); 
            }
        }
        
        __syncthreads();
        
       
        #pragma unroll
        for (int i = 0; i < 32; ++i){
            #pragma unroll
            for(int y = 0; y < 8; ++y){
                int a_row = threadIdx.y * 8 + y;
                reg_a[y] = __half2float(s_a[a_row][i]);
            }
            #pragma unroll
            for(int x = 0; x < 8; ++x){
                int b_col = threadIdx.x * 8 + x;
                reg_b[x] = __half2float(s_b[i][b_col]);
            }
            #pragma unroll
            for(int y = 0; y < 8; ++y){
                #pragma unroll
                for(int x = 0; x < 8; ++x) {
                    acc[y][x] += reg_a[y] * reg_b[x];
                }
            }
        }
        __syncthreads();
    }
    
  
    #pragma unroll
    for (int y = 0; y < 8; ++y) {
        #pragma unroll
        for (int x = 0; x < 8; ++x) {
            int c_global_row = blockIdx.y * 128 + threadIdx.y * 8 + y;
            int c_global_col = blockIdx.x * 128 + threadIdx.x * 8 + x;
            
            if (c_global_row < M && c_global_col < N) {
                int c_idx = c_global_row * N + c_global_col;
                if (beta == 0.0f) {
                    C[c_idx] = __float2half(acc[y][x] * alpha);
                } else {
                    C[c_idx] = __float2half(acc[y][x] * alpha + beta * __half2float(C[c_idx]));
                }
            }
        }
    }
}
#include <mma.h>
using namespace nvcuda;

__global__ void basicGEMM_tensor_cores(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // 1. Padded shared memory (Same as before)
    __shared__ half s_a[128][32 + 8]; 
    __shared__ half s_b[32][128 + 8]; 

    // TODO: Determine Warp ID (0 to 7) and Lane ID (0 to 31)
    int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    int warpId = tid / 32;
    int laneId = tid % 32;

    // 2. Declare WMMA fragments instead of float acc[8][8]
    // A fragment is a special hardware register distributed across the 32 threads
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f); // Initialize to 0

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    for(int k = 0; k < K; k += 32) {
        
        // --- KEEP YOUR EXISTING FLOAT4 VECTORIZED LOADS HERE ---
        // (The 256 threads cooperatively load 128x32 into s_a and 32x128 into s_b)
        // ...
        
        __syncthreads();
        
        // --- NEW TENSOR CORE COMPUTE PHASE ---
        // You loop through the K-dimension of the shared memory tile in steps of 16
        for (int i = 0; i < 32; i += 16) {
            
            // TODO: Calculate the shared memory pointers for this specific warp
            half* s_a_ptr = &s_a[ /* warp's row */ ][i];
            half* s_b_ptr = &s_b[i][ /* warp's col */ ];

            // 3. Load 16x16 tiles from shared memory into the Tensor Core registers
            // The "40" and "136" are the leading dimensions (stride) of your padded shared memory
            wmma::load_matrix_sync(a_frag, s_a_ptr, 32 + 8); 
            wmma::load_matrix_sync(b_frag, s_b_ptr, 128 + 8);

            // 4. Execute the hardware Matrix Multiply-Accumulate
            // This computes 4096 operations in one clock cycle
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }
    
    // --- NEW WRITE-BACK PHASE ---
    // Tensor Cores cannot directly apply alpha/beta math while storing to global memory.
    // TODO: Write the acc_frag back to a new shared memory buffer, 
    // then have the standard threads read it, apply alpha/beta, and write to global C.
}
int main() {
    // 1. Define matrix dimensions
    int M = 4096;
    int N = 4096;
    int K = 4096;
    
    float alpha = 1.0f;
    float beta = 0.0f;

    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(half);

    // 2. Allocate and initialize host memory
    std::vector<half> h_A(M * K, __float2half(1.0f)); // Fill with 1.0
    std::vector<half> h_B(K * N, __float2half(2.0f)); // Fill with 2.0
    std::vector<half> h_C(M * N, __float2half(0.0f));

    // 3. Allocate device memory
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 4. Copy data to device
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // 5. Configure grid and block dimensions
    dim3 threadsPerBlock(32, 32);
    // Ceiling division to handle matrices that aren't perfectly divisible by 32
    dim3 numBlocks((N + 31) / 32, (M + 31) / 32);

    // 6. Launch the kernel
    std::cout << "Launching GEMM Kernel..." << std::endl;
    // 1. Calculate theoretical memory footprint (Read A, Read B, Write C)
    // Calculate theoretical limits
    double bytes = (double)(M * K + K * N + M * N) * sizeof(half);
    double flops = 2.0 * M * N * K;

    // ==========================================
    // 1. BENCHMARK BASELINE KERNEL (32x32)
    // ==========================================
    dim3 threadsBaseline(32, 32);
    dim3 blocksBaseline((N + 31) / 32, (M + 31) / 32);
    
    BENCH("GEMM_Baseline", bytes, flops, 
          basicGEMM<<<blocksBaseline, threadsBaseline>>>(d_A, d_B, d_C, M, N, K, alpha, beta)
    );

    // ==========================================
    // 2. BENCHMARK REGISTER TILING (128x128)
    // ==========================================
    dim3 threadsRegTile(16, 16);
    dim3 blocksRegTile((N + 127) / 128, (M + 127) / 128);
    
    BENCH("GEMM_RegTile", bytes, flops, 
          basicGEMM_register_tiling<<<blocksRegTile, threadsRegTile>>>(d_A, d_B, d_C, M, N, K, alpha, beta)
    );

    // ==========================================
    // 3. BENCHMARK VECTORIZED (128x128, float4)
    // ==========================================
    BENCH("GEMM_VecLoad", bytes, flops, 
          basicGEMM_register_tiling_vectorized<<<blocksRegTile, threadsRegTile>>>(d_A, d_B, d_C, M, N, K, alpha, beta)
    );
    // ==========================================
    // 4. BENCHMARK CUBLAS (Tensor Cores)
    // ==========================================
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Tell cuBLAS it is allowed to use Tensor Cores
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    BENCH("cuBLAS_GEMM", bytes, flops, 
        cublasGemmEx(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, // No transposition
            N, M, K,                  // Swapped dimensions for row-major trick
            &alpha,                   // Alpha scaling factor
            d_B, CUDA_R_16F, N,       // Matrix B comes first! (FP16)
            d_A, CUDA_R_16F, K,       // Matrix A comes second! (FP16)
            &beta,                    // Beta scaling factor
            d_C, CUDA_R_16F, N,       // Matrix C (FP16)
            CUDA_R_32F,               // Compute in FP32 (Float)
            CUBLAS_GEMM_DEFAULT_TENSOR_OP // Optimization algorithm
        )
    );

    cublasDestroy(handle);
    // 7. Copy result back and verify
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    
    // A matrix of 1s multiplied by a matrix of 2s with K=4096 should yield 4096 * 1 * 2 = 8192
    std::cout << "First element result: " << __half2float(h_C[0]) << " (Expected: 8192)" << std::endl;

    // 8. Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}