# cuda-llm

Custom CUDA kernels for fast batch-size-1 LLM decoding on a consumer GPU.

**Goal:** close the gap between vLLM and the hardware's memory-bandwidth limit for Llama-3.2-1B on an RTX 4070 Super.

**Hardware:** RTX 4070 Super (Ada, sm_89, 56 SMs, 504 GB/s, 48 MB L2)
**Stack:** CUDA 13.2, PyTorch 2.14.1+cu132, WSL2 Ubuntu 24.04

## Progress
- [x] Environment set up (nvcc + load_inline working)
- [ ] CUDA fundamentals (GPU-Puzzles, OLCF 1–5)
- [ ] Memory-bound kernels: reduction, softmax, RMSNorm
- [ ] GEMV (fp16, int8)
- [ ] Decode attention
- [ ] End-to-end model + benchmarks vs PyTorch / vLLM

 ncu --section SpeedOfLight --section LaunchStats --section MemoryWorkloadAnalysis  /path

 cr /path
 crp /path