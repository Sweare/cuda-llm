__device__ float reduceMax(float val){
    for(int offset=16;offset>0;offset/=2){
        float neighbor_val = __shfl_down_sync(0xffffffff, val, offset);

        val=fmaxf(val, neighbor_val);
    }
    return val;
}
__device__ float warpReduceSum(float val){
    for(int offset=16;offset>0;offset/=2){
        val+= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
__global__ void softmax(const float* input, float* output, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    __shared__ float warp_sums[8];
    // 1. Load with SAFE negative padding
    float my_val = (i < N) ? input[i] : -1e20f; //0 would be 1
    // 2. Find the max within each 32-thread warp
    max_val=reduceMax(max_val);
    // 3. The 8 warp leaders put their max into the mailbox
    if (laneId == 0) {
        shared_bridge[warpId] = max_val;
    }
    __syncthreads();

    // 4. Warp 0 finds the absolute Block Maximum
    if (warpId == 0) {
        float block_max = (laneId < 8) ? shared_bridge[laneId] : -1e20f;
        block_max = warpReduceMax(block_max);
        
        // 5. THE BROADCAST: Thread 0 leaves the grand winner in slot 0 
        // so everyone can read it after the next barrier
        if (laneId == 0) {
            shared_bridge[0] = block_max; 
        }
    }
    __syncthreads(); // Everyone waits for the grand winner to be posted

    // 6. Every single thread reads the absolute max
    float absolute_max = shared_bridge[0];

    // --- PHASE 1 COMPLETE ---
    
    // TODO: Phase 2 (Subtract max, expf(), and Sum Reduction)
    // 1. Thread-Local Math: Subtract max and calculate e^x. 
    // Notice the out-of-bounds threads are now padded with 0.0f!
    float exp_val = (i < N) ? expf(my_val - absolute_max) : 0.0f;
    float sum_val = warpReduceSum(exp_val);
    if (laneId == 0) {
        shared_bridge[warpId] = sum_val;
    }
    __syncthreads();
    if (warpId == 0) {
        float block_sum = (laneId < 8) ? shared_bridge[laneId] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        
        // 4. THE BROADCAST: Thread 0 posts the denominator for everyone
        if (laneId == 0) {
            shared_bridge[0] = block_sum;
        }
    }
    __syncthreads();
    float absolute_sum = shared_bridge[0];

    if (i < N) {
        output[i] = exp_val / absolute_sum;
    }
}