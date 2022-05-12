# Release 0.4.0-Preview.1 - Setup Biomemap Generator

This release mainly focuses on refining implementations for biomemap generator so it can be integrated into our heightfield generator later.

## Biome implementation

- Enhance memory management for `STPLayer`, remove explicit memory allocation and deletion, instead it's now automatically managed by a new class `STPLayerManager`.
- Enable loading biome settings from `Biome.ini`.
- Add biome map storage into `STPChunk`.
- Reimplement `STPBiomeFactory` for safe parallel biome map generation.
- Implement biome layer chain for demo purpose.
- Integrate layer implementation supplier into `STPBiomeFactory`.
- Add linker function in `STPHeightfieldGenerator` and allow linking `STPBiomeFactory` with it for biome map generation.
- Add biome map generation options to `STPHeightfieldGenerator`, however `STPHeightfieldGenerator` is still not able to generate biome map, integration will yet be implemented until further notice.

## Changes

- Remove `using` in headers to avoid naming conflict.
- Remove `mapOffsetY` from INI entry.
- Change namespace `STPBiome` to `STPDiversity` to resolve naming conflict with class `STPBiome`.
- Rename some files to make our programming style more consistent, such as from `STPBiome_def.h` to `STPBiomeDefine.h`.
- Remove meaningless `const` qualifier for value return type.
- Implement device pointer deleter, and all device memory are now managed by `std::unique_ptr`.