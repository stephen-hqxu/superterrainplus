# Release 0.3.0 - Introducing Free-slip Hydraulic Erosion

This is a update for #15 to address suggestion pointed out in #1.

## New feature

- Water droplet can now freely move to their respective neighbour chunks instead of ending its lifetime early.
- Adding two INI lines `freeSlipX` and `freeSlipZ`, denotes the number of chunk in X and Z direction that will be used as neighbours for free-slip synchronisation.
- Implementing local to global index lookup table to reference neighbour chunks efficiently.
- Implementing interpolation index lookup table to correct edge seams efficiently.
- Update CUDA toolkit version to 11.3 for bug fix stated in #14, improvement in performance and new features, this will be the minimum requirement in all future releases.

## Other improvement

- A major overhaul to kernel launch and host multithread model.
- Remove dedicated normal map cache in memory.
- Integrate heightmap and normalmap into a single texture in rendering buffer.
- Reduce rendering buffer clear buffer size by 20%.
- Merge formatting kernel and normalmap generation kernel into one new kernel, rendering buffer generation kernel, thus cutting down the number of kernel launch and copy both by 20%.
- Pre-cache heightmap into shared memory during normalmap generation.
- Fixing over-allocated memory, improve memory use efficiency.
- `cudaStream_t` is now pooled.
- General programming practice and style improvement.
- Documentation for free-slip erosion.