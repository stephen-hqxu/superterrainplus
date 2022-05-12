# Release 0.5.8 - Miscellaneous Improvement

## Restructure

- Enable free-slip neighbour logic on heightmap generation. `SuperTerrain+` will generate neighbour biomemap before generating the central heightmap, so free-slip biomemap can be passed to heightmap generator for biome interpolation.
- Overhaul the structure of `checkChunk()` in `STPChunkProvider` for better nested neighbour logic.
- Split `STPFreeSlipManager` into three different files, being `STPFreeSlipGenerator` and `STPFreeSlipData` in addition to that. This aims to separate data with implementation.
- Move CUDA context parameter setting from `STPMasterRenderer` to `STPEngineInitialiser`.
- Add `STPKernelMath` in `SuperAlgorithm+Device` for easier programming.

## General fixes and improvement

- Add some checkers in `SuperTerrain+` to prevent user from including internal headers like `STPRainDrop.cuh` which has no exported symbol.
- Remove unused function parameter in `STPChunkProvider`.
- Improvement to documentation.
- Fix a bug when generating biomemap in column-major order which breaks spatial locality of cache.
- Make `STPThreadPool` constructed as an object rather than a dynamic pointer in some classes.