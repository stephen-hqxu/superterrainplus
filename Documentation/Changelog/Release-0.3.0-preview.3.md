# Release 0.3.0-Preview.3 - Various Optimisation to Free-slip Hydraulic Erosion

## Optimisation to free-slip hydraulic erosion (#1)

- Heightmap is now pre-cached into shared memory before normal map calculation for reduced memory latency when the same pixel is accessed up to 9 times.
- Caching from global to shared memory is now done in parallel to utilise as many threads as possible. Improvement to existing caching algorithm to make it smarter such that it can determine if the number of thread in a block is sufficient to cache all data in parallel, and reuse threads whenever necessary.
- Adding checker whenever `loadChunksAsync()` is called and only proceed when there is at least one chunk is not loaded or rendering buffer needs be updated, this avoids unnecessary calls to map rendering buffer and chunk checking.
- Removing FP32 normalmap storage, integrating normalmap and heightmap into one rendering buffer, RGB channel being normalmap and A channel being heightmap. Cut down the number of kernel launch and memory copy both by around 20%.
- Integrating FP32-INT16 converter into normalmap generator, and renamed into `generateRenderingBufferKERNEL()`.
- No longer generate and use global-local index and interpolation table when free-slip hydraulic erosion is disabled.

## General fixes and improvement

- Fixing syntax error from compiler related to built-in thread pool when `emplace_void()` function is called.
- Changing `new` to `make_unique` for better practice.
- Cut down rendering buffer clear buffer size by 20%.
- Update to address changes in dependencies.
- Fixing incorrect serialisation size.
- Remove some redundant codes.