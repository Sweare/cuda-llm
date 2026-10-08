#pragma once
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA_ERR(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } } while (0)

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

#define TIME(label, ...) \
    printf("%-10s %8.3f ms\n", label, time_ms([&] { __VA_ARGS__; }))

#define BENCH(label, bytes, ...) do {                                        \
    for (int _w = 0; _w < 3; _w++) { __VA_ARGS__; }   /* warm-up */          \
    float _ms = time_ms([&] { for (int _r = 0; _r < 20; _r++) { __VA_ARGS__; } }) / 20; \
    printf("%-10s %8.4f ms  %7.1f GB/s\n", label, _ms, (bytes) / _ms / 1e6); \
} while (0)
