# Release 0.8.1 - Device Code Refactoring

## Refactoring

- `SuperDemo+` now terminates when shader compilation fails.

### Remove normalmap from rendering buffer

- Deprecate selective edge copy algorithm introduced in #17. Remove normalmap generation from heightfield generator pipeline. Remove normalmap from rendering buffer.

> Now, heightfield rendering only contains a single channel of formatted heightmap of `unsigned int`, i.e., R16.

- Normal calculation in geometry shader in `SuperDemo+`.
- Change `STPChunkManager` and `STPHeightfieldGenerator` as well as shaders to address the changes.
- Move `Strength` from `STPHeightfieldSetting` to `STPMeshSetting` since normalmap will no longer be generated using our engine. Now mesh surface normal calculation can be done in either geometry shader or fragment shader.
- Remove prototype testing for Sobel filter from `SuperTest+` since normalmap generation is no longer needed.

### Device functions refactor

- Move the following CUDA global functions to a new file `STPHeightfieldKernel`, and use a host function wrapper to launch the device kernel. Note that all host function wrapper comes without the *KERNEL* suffix.
  - From `STPHeightfieldGenerator`
    - curandInitKERNEL()
    - performErosionKERNEL() renamed to hydraulicErosionKERNEL()
    - generateRenderingBufferKERNEL() renamed to texture32Fto16KERNEL()
  - From `STPFreeSlipGenerator`
    - initGlobalLocalIndexKERNEL()
- Hence, all instances of global functions in the original classes mentioned above are removed. This greatly reduce the code complexity.
- Relocate the following files from /GPGPU directory to /World/Chunk:
  - STPHeightfieldGenerator
  - STPDiversityGenerator
  - All files under /FreeSlip
- Rename the suffix of the following files from *.cuh*/*.cu* to *.h*/*.cpp*
  - STPHeightfieldGenerator
  - STPFreeSlipGenerator