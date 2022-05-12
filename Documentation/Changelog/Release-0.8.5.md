# Release 0.8.5 - Setup Splatmap Generation

## Splatmap generator

- Move nested structure `STPSplatDatabase` from `STPTextureFactory` to `STPTextureInformation` and rename it to `STPSplatRuleDatabase`.
- Add `lower_bound()` function in `STPKernelMath` similar to the standard library version of this function.
- Setup `STPTextureSplatRuleWrapper` in `SuperAlgorithm+Device` as a helper class to access splat rules.

### Setup SuperDemo+ splatmap generation

- Add `STPCommonCompiler` as a reusable compiler options loader. Remove compiler options setup in `STPBiomefieldGenerator`.
- Move all runtime-compiled scripts to `Script` directory in `SuperDemo+` root. Rename filename suffix to *.cu* for consistency.
- Create `STPSplatmapGenerator` and setup splatmap generation routines. Also setup runtime script for device generation.

## General fixes and improvement

- Add new static utility functions `getChunkCoordinate()`, `calcChunkMapOffset()` and `getLocalChunkCoordinate()` to retrieve chunk coordinate and map offset in `STPChunk`.
- Move smart memory and memory pool classes to a subdirectory Memory.
- Change the way to retrieve `cudaSteam_t` in `STPSmartStream`, instead of using implicit cast it now uses dereference operator.
  - Declared `noexcept` for that function.
- Change implicit cast to free-slip location to a function `where()` in `STPFreeSlipTextureBuffer`.
- Add OpenGL compatibility checking in `STPEngineInitialiser`.
- Replace memory block container in `STPMemoryPool` from `std::unordered_map` to a sorted array, because we expect the number of entry within the memory pool to be very small (no more than 10) and hash table is unnecessary.
- Remove unnecessary cast to `uintptr_t` when passing runtime compiler flags.
- Add invalid data checking during construction of `STPTextureFactory` and before any memory is allocated.
- Remove default value from `STPPermutation` so CUDA can initialise it in constant memory.
- Add some contents to `README.md`.

### STPChunkManager

- Fix an undefined behaviour, change register flag for `cudaGraphicsGLRegisterImage()` from `cudaGraphicsRegisterFlagsWriteDiscard` to `cudaGraphicsRegisterFlagsNone`, because our pipeline only updates a portion of the whole texture each time rather than asking CUDA to discard the entire content.
  - A resource mapping flag `cudaGraphicsResourceSetMapFlags()` is also added with `cudaGraphicsMapFlagsNone` set.
- Use `unsigned int` rather than `int` to represent chunk ID.
- Refactor setting up texture in the constructor.
- Group `cudaArray_t` into a structure to reduce the number of function argument.
- Refactor copying buffer to GL texture using lambda functions.