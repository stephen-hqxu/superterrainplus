# Release 0.4.0-Preview.2 - Enable Biomemap Generation

In this release our heightfield generator can generate biome map and store the map for each chunk. Later we will integrate heightmap generation with biomemap to create a multi-biome procedural world.

## Highlights

- Break down chunk management engines into individuals, and remove automatic implicit constructions. For example previously instantiating `STPChunkProvider` will have `STPHeightfieldGenerator` instantiated automatically, now `STPHeightfieldGenerator` is passed as a reference into the constructor call of `STPChunkProvider`.
- All parameters passed to above said class are now by reference instead of copy, such that if one wishes to keep using low-level API call (constructing each component individually), all components and settings need to be kept alive until the engine is destroyed.
- Add `STPWorldManager` as a high-level API for easy initialisation of all generation engine components.

> The purpose of doing this is to create some level of abstraction for each component, such that specific functions from each stage of the pipeline can be called directly from external environment. This will be particularly useful for setting up biome generator.

- Allow attaching a concrete instance of `STPBiomeFactory` to `STPWorldManager`.
- Allow heightfield generator linked with biome factory for biomemap generation.
- Enable non-blocking biomemap generation.
- Load biomemap into OpenGL buffer and enable biomemap to be visualised in fragment shader.

## General improvement and fixes

- Add `const` to some member functions of chunk engine components so they can be easily called by the returning reference from `STPWorldManager`.
- Rename `sample_cached()` in `STPLayer` to `retrieve()` to avoid confusion and accidental call to `sample()` when writing concrete classes of `STPLayer`.
- Rename `create()` in `STPLayerManager` to `insert()`.
- Remove seed and salt argument in the `insert()` above, allowing more abstract constructor of derived `STPLayer`