#include "common.cuh"

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col=blockIdx.x*blockDim.x + threadIdx.x;
    int row= blockIdx.y*blockDim.y+ threadIdx.y;
    
    if(row<rows && col<cols){
        output[col*rows+row]=input[row*cols+col];
    }

    }