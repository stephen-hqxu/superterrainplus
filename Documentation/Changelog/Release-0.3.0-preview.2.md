# Release 0.3.0-Preview.2 - Improving Free-slip Hydraulic Erosion

**CUDA compiler version has been updated to 11.3, future release of SuperTerrain+ will have CUDA 11.3 as minimum requirement due to compatibility restriction**

## Improvement to free-slip hydraulic erosion algorithm (#1)

- Erosion brush indices are now correctly computed with free-slip range instead of heightmap size.
- Re-program the structure of `generateHeightmap()` function (now is `operator()`) to make it more structured and execution-efficient.
- Normal map generation and formation are now done to all neighbours after erosion such that the rendering buffer is correct.
- Seam between chunks is now fixed with post-erosion edge interpolation with interpolation lookup table which is generated during class initialisation.
- A major overhaul to the thread model in host side.

### Plan

- Cutting down kernel launch by using device subroutine, i.e., group edge interpolation, normal map generation and formation into one kernel launch, and use flag to identify which operation to perform.
- Change heightfield generator `operator()` so it can be more flexible and adaptive. For example in current build function forces user to compute normal map and formation in a free-slip manner if free-slip dimension has been defined when initialising the class. A potential solution is to let user input free-slip dimension at function call instead of storing the parameters in the class object.
- Adding support for async device memory management (`cudaMallocAsync` and `cudaFreeAsync`) and built-in memory pool introduced in CUDA 11.3
- Cache repeated global memory access on device into shared memory for further performance enhancement.
- Simplify multithread model.

> Please refer to the documentation in source code and issue link (https://github.com/stephen-hqxu/superterrainplus/issues/1#issuecomment-868819599) for more developer discussions regarding this lookup table.

## Other improvements

- Fully addressed issue identified as in #14, mitigated by updating `nvcc` to 11.3.
- Reducing the number of memory pool, memory usage now is more efficient.
- Fixing over-allocated device and page-locked memory.
- Refactoring memory copy between device and host.
- Optimise CUDA memory R/W access.
- Constant numbers are computed inside kernel and stored in shared memory to improve performance.
- Erosion brush indices and weights are now cached into shared memory.
- CUDA kernel streams are pooled and reused to reduce stream creation/destroy overhead.
- Fixing data incoherence in multithreaded environment on host by force reloading updated chunks.

> There is noticeable performance gain by applying those improvements, which is good.