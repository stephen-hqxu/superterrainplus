# Release 0.3.4 - Texture Seams Elimination

## What's new

- Merging layered rendering buffer into a single rendering buffer, such that texture filtering and edge interpolation that fixes terrain seams are now automatically performed by OpenGL. (#17)
- Completely eliminate normal map edge artefact by selectively pre-copying the edge from previous rendering buffers selectively.
- Post erosion interpolation has been removed completely.
- Implementing built-in memory pool introduced in CUDA 11.2.
- Group rendering buffer updates for all required chunks into a single API call/kernel launch.
- Adding error checking for all CUDA API calls.
- CUDA kernel launch configurations are now determined dynamically, also grid size calculation has been improved to address the case when texture size or raindrop count etc. are not a whole number multiple of block size. (#16)

## Fixes and improvements

- Fix an incorrect indexing calculation during rendering buffer generation.
- Remove some redundant `const` keyword when pass by value.
- Factorise `cudaStream_t` usage, program reuses stream more efficiently.
- Combine some initialisation functions into the class constructor call so they don't need to be called explicitly.
- Replace `std::list` with `std::vector` where applicable.