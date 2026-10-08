#include "common.cuh"
#include <cublas_v2.h>
#include <chrono>

__global__ void matMul(const float* A, const float* B, float* C, int N){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col<N){
        float sum=0.0f;

        for(int k=0;k<N;k++){
            sum+=A[row*N+k]*B[k*N+col];
        }
        C[row*N+col]=sum;
    }
}
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += A[r * N + k] * B[k * N + c];
            C[r * N + c] = s;
        }
}
void fill_random(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;   // [-1, 1]
}
void run(int N, bool check) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes), *h_C = (float*)malloc(bytes);
    fill_random(h_A, N * N); fill_random(h_B, N * N);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERR(cudaMalloc(&d_A, bytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_B, bytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_C, bytes));
    CHECK_CUDA_ERR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    // ---- 1. correctness ----
    if (check) {
        matMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        float* ref = (float*)malloc(bytes);

        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_cpu(h_A, h_B, ref, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("N=%d  CPU   : %8.2f ms  %8.1f GFLOPS\n", N, cpu_ms, 2.0 * N * N * N / (cpu_ms * 1e6));
        float max_err = 0.0f;
        for (int i = 0; i < N * N; i++) max_err = fmaxf(max_err, fabsf(h_C[i] - ref[i]));
        printf("N=%d  correctness: max error = %.2e  %s\n", N, max_err, max_err < 1e-2f ? "PASS" : "FAIL");
        free(ref);
    }

    // ---- 2. performance: your kernel vs cuBLAS ----
    double flops = 2.0 * N * N * N;   // one multiply + one add per term

    for (int w = 0; w < 2; w++) matMul<<<blocks, threads>>>(d_A, d_B, d_C, N);   // warm-up
    float ms_naive = time_ms([&] { for (int r = 0; r < 5; r++) matMul<<<blocks, threads>>>(d_A, d_B, d_C, N); }) / 5;

    cublasHandle_t handle; cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major; computing B·A in column-major gives A·B in row-major
    auto gemm = [&] { cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                  &alpha, d_B, N, d_A, N, &beta, d_C, N); };
    for (int w = 0; w < 2; w++) gemm();
    float ms_cublas = time_ms([&] { for (int r = 0; r < 5; r++) gemm(); }) / 5;
    cublasDestroy(handle);

    printf("N=%d  naive : %8.2f ms  %8.1f GFLOPS\n", N, ms_naive,  flops / (ms_naive  * 1e6));
    printf("N=%d  cuBLAS: %8.2f ms  %8.1f GFLOPS  (naive reaches %.1f%% of cuBLAS)\n",
           N, ms_cublas, flops / (ms_cublas * 1e6), 100.0 * ms_cublas / ms_naive);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

int main() {
    run(512, true);     // small: check correctness against the CPU
    run(4096, false);   // large: benchmark (CPU check would take minutes)
    auto t0 = std::chrono::high_resolution_clock::now();
    
    return 0;
}