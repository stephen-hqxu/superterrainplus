# Release 0.6.7 - Better Interpolation and Project

## Better smooth biome transition

- Simplify the data structure model for `SuperAlgorithm+Device`. Include data structure headers in each main class header to eliminate the need of declaring opaque pointers in runtime device script.

> Pros: It has cut down register usage on GPU and significantly reduced the programming difficulty.
Cons: NVRTC needs to know the include directory of `SuperTerrain+` Core engine due to the use of `STPBiomeDefine` in `STPSingleHistogram`.

- Simplify single histogram data being sent to device, instead of sending a pointers wrapped over a number of pointers, the structure which contains those pointers is sent directly.
- Add custom allocator for `STPArrayList`.
- Fix error thrown during pinned memory allocation in histogram buffer.

## Build system change

- Remove deprecated CMAKE options:
  - STP_CUDA_ARCH: now use CMAKE built-in CUDA arch variable.
  - STP_ENGINE_BUILD_SHARED: `SuperTerrain+` will now always be built into shared library
  - STP_DEVICELINK_STATIC: no longer useful when the engine is always built into shared.
- Move `STPCoreDefine` into directory `CoreInterface` from `Template`.
- Merge `SuperError+` into `SuperTerrain+` Utility folder, and remove `SuperError+` build target.

## General improvement and fixes

- Add `interpolationRadius` in `Biome.ini` for dynamic configuration.
- Add core include directory information into `STPAlgorithmDeviceInfo`.
- Use placement new to initialise data allocated from memory pool instead of using `memset`.