# Release 0.4.0 - New Biome and Height Generator

## Biomemap generator

- Abandon linked list structure of `STPLayer`. Add a separate master manager `STPLayerManager`.
- Re-implement `STPBiomeFactory` with a better programmable interface, as well as concurrent heightmap generation and memory management.
- Integrate demo biome generator into the generator pipeline.
- Import settings from `Biome.ini`.
- Biomemap can now be stored in `STPChunk` and loaded into OpenGL rendering buffer.

## Generalised heightmap generator

- Add `STPDiversityGenerator` which contains some high-level helper functions for simple runtime complication.
- Allow programmable heightmap generation, such that user can develop their biome-specific algorithms.
- Allow runtime-compiled script for heightmap generator. Current heightmap generation pipeline will become deprecated and change to programmable pipeline in future release.

## Other fixes and improvement

- Move some include into source code.
- Change some names for consistent programming style and conflict resolution.
- Deprecate INI entry `mapOffsetY`.

> Also note that all heightmap generator parameters like *octave* and *persistence* will be deprecated in future release.

- Separate chunk management engine and heightfield generator, so creating renderer won't lead to a chained creation of `STPChunkManager`, `STPChunkProvider` and `STPHeightfieldGenerator`. Instead, individual parts need to be created by user and linked to the depended program by reference. This gives user better controls on whether they want to share it with other pipeline, for example.
- Add a high-level management unit `STPWorldManager` in case developer doesn't need such control mentioned above and wish to initialise the generator with ease.

> For more technical details please refer to the pull request page #19.