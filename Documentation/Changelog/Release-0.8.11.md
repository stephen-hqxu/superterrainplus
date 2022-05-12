# Release 0.8.11 - Multi-Biome Texture Splatting

## Generator pipeline overhaul

This release provides solution to #30. `STPChunkManager` and `STPChunkProvider` are deprecated and removed from the engine, and replaced by a single `STPWorldPipeline` that functions exactly the same as them. However there are a few changes.

- Rename some functions to make them more compact.
- Chunk can be retrieved in one function call, instead of previous two (check first then get).

## Apply texture to splatmap

- Allow retrieval of texture in `STPTextureFactory`; allow retrieving texture object using group ID.
- Add `STPSplatTextureDatabase` for storing pointers to texture data in `STPTextureInformation`.
- Setup texture parameters for splatting texture objects. This is currently done in `STPWorldManager`.
- Setup texture system in `STPProcedural2DINF`.
- `SuperDemo+` can now render terrain with texture using splatmap. Improve UV coordinate system to make sure texture is stable when chunk position is updated.

### Improvement to splatmap generator

- All splatmap generator parameters are put inside `STPSplatmapGenerator.cu` as constants instead of importing from INI file.
- Add simplex noise to terrain height to make the region boundary more natural.
- Change texture coordinate during splatmap generation from normalised UV to original pixel coordinate. Normalised UV causes floating point precision issues such that splatmap is unstable when rendered chunk changes world position. It also improves generation efficiency significantly.

> We will be optimising the texture system later.

## General fixes and improvement

- Fix a potential illegal memory access where texture and surface objects used during splatmap generation are deleted before kernel execution is finished.
- Change the way to retrieve permutation table in `STPPermutationGenerator` from calling `operator()` to `operator*`.
- Fix an incorrect bound when looping through gradient registry in the device splat rule wrapper.
- Simplex noise permutation table is now shared across all runtime compiled program, located in `STPCommonCompiler`.
- Add `const` functions in `STPChunk` for retrieving various maps.