#include "common.cuh"

__global__ void vadd(const int* a,const int* b, int* c, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i<n){
        c[i]=a[i]+b[i];

    }
}

int main(){
    int n=10;
    size_t size= n*sizeof(int);
    int* a_h = (int*)malloc(size);
    int* b_h = (int*)malloc(size);
    int* c_h = (int*)malloc(size);

    // Initialize host arrays so you aren't adding uninitialized garbage memory
    for (int i = 0; i < n; i++) {
        a_h[i] = i;
        b_h[i] = i * 2;
    }

    int* a_d;
    cudaMalloc(&a_d, size);
    int* b_d;
    cudaMalloc(&b_d, size);
    int* c_d;
    cudaMalloc(&c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    int threadsPerBlock=256;
    int blocksPerGrid=(n + threadsPerBlock-1) / threadsPerBlock;

    vadd<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c_h, c_d,size, cudaMemcpyDeviceToHost);

    // Print results to verify (uses your <cstdio> from common.cuh)
    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a_h[i], b_h[i], c_h[i]);
    }
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}