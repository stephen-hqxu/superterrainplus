# Release 0.12.11 - Patch and Optimisation

## General fixes and improvement

- Update `SIMPLE` to version 3.0.
- Fix an incorrect use of comma operator during initialisation of `const SIStorage&` in `STPStart` which creates unnecessary data copies.
- Fix an incorrect use of `inline` in the following files which causes link-time undefined reference when compile under Release build:
  - `STPSmartDeviceMemory`
  - `STPScreen`
  - `STPScenePipeline`
- For `STPShaderManager`, improve shader macro defining with `std::string_view`. Add support for detecting `#define` with leading and in-between white space.
- All use of `std::queue` are changed to use `std::deque` as container instead of `std::list`.
- Deprecate and remove `STPChunkStorage` because it is just a useless thin wrapper over `std::unordered_map`.
- Change the declaration order for `STPThreadPool` in `STPWorldPipeline` and `STPSingleHistogramFilter` to ensure the correct destruction order and avoid illegal memory access.
- For `STPOpenGL.h`, `STPuint64` is now expressed as `uint64_t` to make it compatible with system which uses different specification to define `GLuint64`.
- Fix an error on some compilers where:
  - `sqrt()` is not found in `STPErosionBrushGenerator` by including correspond maths header.
  - `memset` is not found in `STPLayerCache` by including *cstring*.
  - `std::shuffle` is not found in `STPPermutationGenerator` by including *algorithm*.
  - `std::list` is not found in `STPShaderManager` and `STPThreadPool`.
- Remove useless `typename` declaration which causes errors for some compilers, from:
  - `STPSmartDeviceMemory.tpp`
  - `STPMemoryPool.h`
  - `STPFreeSlipTextureBuffer.h` and `STPFreeSlipTextureBuffer.cpp`
  - `STPSingleHistogramFilter.cpp`
  - `STPSceneObject.h`
- Remove extra namespace qualification for some functions in `STPScenePipeline`.
- Launch buffer size in `STPBiomefieldGenerator` and `STPSplatmapGenerator` now uses `sizeof` to determine rather than using hardcoded values because pointer size on different platform might not be always 8 byte.

### STPWorldPipeline

- Fix an undefined behaviour where a temporary returned vector is used in a range-based for loop.
- Fix a race condition where chunk might be used while it is being occupied.

To avoid running into the nasty race condition again, we introduce `STPChunk::STPMapVisitor` which has two operation modes: shared and unique. It works like a shared/unique mutex in the standard library, the only difference is rather than stalling threads when mutex condition is unmet, exception will be generated. This attempts to prevent developers from making this mistake in the future again.

### STPTextureDefinitionLanguage

- Fix a bug where a `std::string` is created from calling the `data()` function in `std::string_view` which basically copies the entire chunk of memory for every number parsed and results in significant memory overhead.
- Simplify `readNumber()` and `readString()` functions. Remove repeated string read.