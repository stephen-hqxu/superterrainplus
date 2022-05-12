# Release 0.5.0 - File System Overhaul

This release aims to replace VS 2019 build system with CMAKE 3 due to some unforeseen bugs related to Visual Studio, and improve cross-platform compatibility. Also remap the current file hierarchy to separate the actual engine with the demo program.

Before building system with CMake, unzip `STPExtras.zip` in the root project directory.

## Overhaul to build system

- Deprecate VS 2019 build system, instead SuperTerrain+ will use CMake 3.18. VS 2019 solution can however be generated.
- Separate source code and header.
- Split `STPDeviceErrorHandler` into a separate shared library target `SuperError+`.
- Split `STPSimplexNoise` and `STPPermutationsGenerator` into a separate static library target `SuperAlgorithm+` to allow the newly introduced runtime compiled heightmap generator script to load noise library directly during runtime.
- Completely split `SuperTerrain+` main engine from the demo program `SuperDemo+`.
- Add header files to build dependencies so IDE like Visual Studio can see it.
- Add a template file `STPCoreDefine` for easier runtime filename referencing.

## Rename

- Change `Helpers` directory in `SuperTerrain+` to `Utility`.
- Change `Settings` directory in `SuperTerrain+` to `Environment`.
- Change `Biome` directory in `SuperTerrain+` to `Diversity`.
- Change namespace `STPSettings` to `STPEnvironment` to avoid name conflict .
- Change `STPPermutationsGenerator` to `STPPermutationGenerator`.