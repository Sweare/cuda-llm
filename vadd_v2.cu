#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA_ERR(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at line %d\n", \
    cudaGetErrorString(err), __LINE__); return 1; } } while (0)

#define BENCH(label, bytes, ...) do {                                        \
    for (int _w = 0; _w < 3; _w++) { __VA_ARGS__; }   /* warm-up */          \
    float _ms = time_ms([&] { for (int _r = 0; _r < 20; _r++) { __VA_ARGS__; } }) / 20; \
    printf("%-10s %8.4f ms  %7.1f GB/s\n", label, _ms, (bytes) / _ms / 1e6); \
} while (0)

#define TIME(label, ...) \
    printf("%-10s %8.3f ms\n", label, time_ms([&] { __VA_ARGS__; }))

template <typename F>
float time_ms(F&& fn) {
    cudaEvent_t s, e; float ms = 0.0f;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    fn();
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}



__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}


int main() {
    int n = 1 << 26;   // 64M floats → 256 MB per array, 768 MB total
    size_t sz = n * sizeof(float);

    float *h_a = (float*)malloc(sz), *h_b = (float*)malloc(sz), *h_c = (float*)malloc(sz);
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *a, *b, *c;
    CHECK_CUDA_ERR(cudaMalloc(&a, sz));
    CHECK_CUDA_ERR(cudaMalloc(&b, sz));
    CHECK_CUDA_ERR(cudaMalloc(&c, sz));



    // --- time the host-to-device copies ---
  
    TIME("H2D", cudaMemcpy(a, h_a, sz, cudaMemcpyHostToDevice);
                cudaMemcpy(b, h_b, sz, cudaMemcpyHostToDevice));




    BENCH("vadd", 3 * sz, vadd<<<(n + 255) / 256, 256>>>(a, b, c, n));

    CHECK_CUDA_ERR(cudaGetLastError());   // catches bad kernel launches

    TIME("D2H",    cudaMemcpy(h_c, c, sz, cudaMemcpyDeviceToHost));
    printf("c[0] = %f, c[n-1] = %f (expect 3.0)\n", h_c[0], h_c[n-1]);


    free(h_a); free(h_b); free(h_c);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}