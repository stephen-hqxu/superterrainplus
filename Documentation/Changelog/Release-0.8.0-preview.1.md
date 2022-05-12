# Release 0.8.0-Preview.1 - Data Structure for Texture Utility

## Biome-specific texture building system

### STPTextureDatabase

STPTextureDatabase is a non-owning storage of collection of texture data and information.

As a whole, texture ID and texture type acts as a compound primary key to uniquely identify the texture data. Internally, texture data with the same property will be grouped into a structure `STPTextureGroup`. This is beneficial for later packing texture with the same format into the same OpenGL texture array.

### STPTextureSplatBuilder

A simple utility that allows each biome to have different texture. Texture can have different splatting rules within the biome on the terrain, and currently the builder allows altitude and gradient splatting.

### STPTextureFactory (WIP)

A generator class that takes texture database and splat builder, combine them in new data structures such that device and shader can make use of those information efficiently to generate terrain texture splatmap.

We have the data structure ready for use, data generation will be coming later...

## General fixes and improvement

- Update `SuperTest+` to support the latest release of `Catch2`:
  - As of https://github.com/catchorg/Catch2/commit/3f8cae8025f4f5e804383f44bb393a63bcef90a4 and https://github.com/catchorg/Catch2/commit/f02c2678a1a891f9577712fe160c1f6f3baef3a8
    - Update `STPConsoleReporter` and replace most `std::string` with `std::string_view` to eliminate unnecessary memory operations.
    - Allow reporter to use `Catch::StringRef` and convert it to `std::string_view`.
    - Reporter now indents the test section name based on the nested depth.
  - As of https://github.com/catchorg/Catch2/commit/426954032f263754d2fff4cffce8552e7371965e
    - Make some modification to `STPTestRTC` to support the renamed matcher.
    - Change floating point comparison in `STPTestPermutation` with Catch build-in floating point matchers.
- Replace most `emplace()` for map containers with `try_emplace()`.
- Declare all move constructor as `noexcept`.
- Change type of PV matrix mapping from `char*` to `unsigned char*` in `STPMasterRenderer`.
- `STPChunkManager` now sync `cudaStream_t` when `cudaGraphicsUnmapResources()` instead of doing it explicitly during async loading. Also stream is now no longer sync after each rendering buffer clear operation. This should result in better device workflow overlap.
- Extract the hash combine algorithm from `STPChunkStorage` into a new class `STPHashCombine`.

### Improvement to STPSingleHistogramFilter

- Simplify loops with functions from standard library algorithm header.
- Change the reallocation condition check from *greater or equal* than the last element to *greater*.
- Add a return type to `insert_back_n()`, which now returns the iterator to the last inserted element to eliminate the need of indexing the element.
- Replace C standard library functions with C++ version, for example `memmove()` to `std::copy()`.