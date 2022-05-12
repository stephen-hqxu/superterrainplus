# Release 0.8.9 - Texture Splatmap Generation

## Enabling splatmap generation in SuperDemo+

In this release we enable the pipeline for texture splatmap generation and it can be visualised during runtime. We will be focusing on applying the texture according to the splatmap and parameter tweaking later.

- Setup external texture loading which takes place in a new sub-class `STPWorldSplattingAgent` in `STPWorldManager` in the demo program.
  - Setup texture loading and texture database in this new sub-class.
- Augment `STPTextureStorage` in the demo program for smart texture memory management.
- Enable parallel texture loading from file system in `STPSkyRenderer` and `STPWorldManager`.
- Add `STPCommonGenerator.cu` in the demo program as a data holder for RTC, allowing all generator scripts to be linked as a single program.

### SuperTerrain+ Texture Definition Language

- Introduce TDL as a handy custom script for defining texture splatting rules.
- Refactor the lexer and parser with error handling improvement.
- Add a new exception type `STPInvalidSyntax` for any lexing and parsing error.
- Fix a problem of calling `front()` from an empty `std::string_view`. This is because neither `std::string` nor `std::string_view` includes null terminator.

## General fixes and improvement

- Eliminate unused texture type when generating texture information in `STPTextureFactory`.
- Remove `__constant__` qualifier for constant values in `STPSimplexNoise` device kernel.
- Refactor all runtime compilers and merge all of them into a single linkable program in `STPCommonCompiler`.
- Change `rehash()` to `reserve()` for `std::unordered_map` used in `STPTextureFactory`.
- Improve robustness of `STPSingleHistogramFilter`, the last working thread will guarantee to finish all remaining tasks if the texture dimension is not divisible by degree of concurrency. Change the threading strategy so `STPSingleHistogramFilter` is now thread safe and results in better concurrency with less waiting. Remove critical section in the demo program when calling histogram filter.
- Move `Exception` to `SuperTerrain+` root directory.
- Refactor functions to attach inherited object and remove unused functions from `STPWorldManager`.
- Fix an incorrectly set launch configuration in `STPSplatmapGenerator`.
- Resolve #28 by completely removing stream callback to deallocate host memory for `STPFreeSlipTextureBuffer` and `STPBiomefieldGenerator`. It's now simply done using a stream sync call.