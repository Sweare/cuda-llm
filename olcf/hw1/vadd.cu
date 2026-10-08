#include <stdio.h>
#include "common.cuh"



__global__ void vadd(const float* a, float* b,float* c, int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;


    c[i]= (i<n) ? a[i]+b[i]: 0.0;

}

int main(){
    int n = 1 << 26;                     // << 20 == 1M elements
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }


    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_c, bytes));
    TIME("H2D",CHECK_CUDA_ERR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    BENCH("vadd",3*bytes,vadd<<<blocks, threads>>>(d_a, d_b, d_c, n));
    CHECK_CUDA_ERR(cudaGetLastError());

    CHECK_CUDA_ERR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++)
        if (h_c[i] != 3.0f) { printf("Wrong at %d: %f\n", i, h_c[i]); return 1; }
    printf("All %d results correct\n", n);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}