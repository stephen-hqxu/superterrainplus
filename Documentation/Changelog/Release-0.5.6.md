# Release 0.5.6 - Runtime Multi-Biome Generation

## Runtime multi-biome heightfield generation in demo program

- Implement runtime-compiled heightfield generation in `SuperDemo+` with `STPDiversityGeneratorRTC`.
- Separate `STPBiomeSettings` into another structure `STPBiomeProperty` which is dedicated to be used by runtime compiler to minimise include directories and compile time.
- Split `STPAlgorithmDefine` into another file `STPAlgorithmDeviceInfo...`. The former one contains shared library export symbols while the latter one contains path to `SuperAlgorithm+Device` library and include. This reduces file conflicts and remove any need to hard-code the filename based on different configuration.
- Implement multi-biome heightfield generation, generation settings are picked from a biome lookup table indexed by biomemap.

> There are 2 biomes implemented in the demo program, being ocean and plains, and the parameters are not the best. The overall terrain looks coarse, we will be looking at biome edge interpolation later.

## General fixes and improvement

- Add guard to check for duplicate source name when compiling source with `compileSource()`.
- Fix a case when calling `discardSource()` the actual `nvrtcProgram` is not deleted, causing memory leaks.
- Improve error handling on `STPDiversityGeneratorRTC` and throw exception when necessary.
- Name expressions are now cached into complication database in `STPDiversityGeneratorRTC` so they can be retrieved later without need to re-typing everything.
- Change the template `STPAlgorithmDefine.h.in` to output the path to device include and library.
- Add a utility in `STPDiversityGeneratorRTC` for reading source code from local file.
- Move CMake configure_file output to binary directory (#20).
- Remove redundant gradient table shuffle. Note that the generation result will be different from the previous implementation.
- Increase the camera height on the demo application.