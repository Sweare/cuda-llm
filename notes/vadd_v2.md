411.1 GB/s ÷ 504 GB/s ≈ 82% of peak. That’s your first honest bandwidth measurement, and a solid result for a simple kernel.

Measurement	Data	Time	Bandwidth
vadd kernel	768 MB	1.96 ms	411 GB/s (82% of peak) ✅
H2D copies	512 MB	28.8 ms	~17.8 GB/s
D2H copy	256 MB	258 ms	~1 GB/s ⚠️

What these numbers tell you:

The kernel is near the realistic ceiling. Nobody reaches 100% of the advertised 504 GB/s. Real hardware loses some to memory refresh and overhead, so roughly 85–92% is usually the practical maximum. To find your card’s real ceiling, time a device-to-device cudaMemcpy of a large buffer (bandwidth = 2 × bytes ÷ time, since it reads and writes). That’s a better target than 504.
H2D improved from about 10 to 17.8 GB/s just because the transfer is bigger. Large copies spread out the fixed overhead per call.
D2H is now terrible, around 1 GB/s. That’s the untouched-malloc problem, much worse now: 256 MB of pages being allocated one by one during the copy. Pinned memory (cudaMallocHost) should fix it dramatically.

Your full list of first findings from today:

Managed memory on WSL: 330 µs with 0% DRAM, because the data stayed in CPU RAM.
Small benchmarks lie: 1412 GB/s came from the L2 cache, not VRAM.
Honest kernel bandwidth: 411 GB/s, 82% of peak.
Copies over PCIe are 25–400x slower than VRAM, so data has to stay on the GPU.
Untouched malloc memory makes copies to the CPU extremely slow.