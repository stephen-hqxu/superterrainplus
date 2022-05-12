# Release 0.8.0 - Texture Splating Utility

In this release we introduces various tools for texturing which will be implemented in the later release, as well as some general quality-of-life improvements to our engine.

> See developer notes in #29 to learn more about rule-based biome-dependent texture splatting 

## Texture

### STPTextureDatabase

STPTextureDatabase is a non-owning storage of collection of texture data and information.

As discussed in the related PR, textures are managed into *texture*, *group* and *map*. Each texture contains multiple maps of different types, and one map belongs to a group which contains properties of all maps.

### STPTextureSplatBuilder

Part of `STPTextureDatabase`. A simple utility that allows each biome to have different texture. Texture can have different splatting rules within the biome on the terrain, and currently the builder allows altitude and gradient splatting. Each rule maps to a texture which allow implementation to render the texture for that active region.

### STPDatabaseView

Part of `STPTextureDatabase`. It allows querying large result sets from database for efficient data processing by `STPTextureFactory`.

### STPTextureFactory

A generator class that takes a texture database, making some batch queries and convert all texture and rules into new data structures such that device and shader can make use of those information efficiently to generate terrain texture splatmap.

## New

- Introduce `STPHashCombine` for mixing up hash values. It was previously located in `STPChunkStorage`.
- Introduce `STPRuntimeCompilable` as a high-level wrapper to NVRTC APIs, it was previously part of `STPDiversityGeneratorRTC` which has been removed in this update. To keep using JIT compilation feature for diversity generator client must inherit both `STPDiversityGenerator` and `STPRuntimeCompilable`.
- `SQLite` is now one of the middleware of `SuperTerrain+`. Introduce `STPSQLite.h` as a compatibility include header.
- Introduce `STPOpenGL` as a compatibility header to `glad.h`.
- New exception type `STPDatabaseError`.
- New error handling functionality for errors from `SQLite`.

## General fixes and improvements

- Update unit test system to https://github.com/catchorg/Catch2/commit/48a889859bca45ee2c5e5064199c1e5b4b3e00cb
- Replace `std::string` usage in `STPConsoleReporter` with `std::string_view`.
- Refine reallocation condition, array insertion function and memory operations in `STPSingleHistogramFilter`, making it more efficient.
- Replace static-only classes with namespaces.
- Update `README` and `LICENSE`.
- Declare `noexcept` for all copy/move constructors/assignment operators.
- Improve CUDA stream sync. behaviour in `STPChunkManager`.