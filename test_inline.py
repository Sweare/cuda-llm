import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
__global__ void add10_kernel(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + 10.0f;
}
torch::Tensor add10(torch::Tensor a) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    add10_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
'''
cpp_src = "torch::Tensor add10(torch::Tensor a);"

mod = load_inline(name="add10", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["add10"], extra_cuda_cflags=["-O3"], verbose=True)

a = torch.arange(8, device="cuda", dtype=torch.float32)
print(mod.add10(a))
print("match:", torch.allclose(mod.add10(a), a + 10))
