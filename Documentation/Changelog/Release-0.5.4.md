# Release 0.5.4 - Better Algorithm Library for Runtime Compilation

`SuperDemo+` is temporarily disabled for this release due to the high amount of updates to the main engine, we will make it work in the next release.

## Runtime compiler improvement

- Separate `SuperAlgorithm+` into two libraries, being `SuperAlgorithm+Host` for host initialisation, and `SuperAlgorithm+Device` for device execution.

> For static complication, both libraries need to be included with the application. For runtime complication, only `SuperAlgorithm+Host` is required, and add `SuperAlgorithm+Device` to library dependencies of JIT linker.

- Make constants compile-time initialised to eliminate the need of calling initialisation function in `STPSimplexNoise`.

## Shared library

- `SuperTerrain+` engine can now support shared library build.
- Add `STPEngineInitialiser`. If shared library is used or there's no other OpenGL contexts bounded to GLAD being created in the application regardlessly, underlying functions must be called before calling other STP APIs to initialise OpenGL and CUDA context.
- Add CMake option to enable device link on core engine static library build.

## General improvement and fixes

- Change all namespaces in `SuperDemo+` to `STPDemo` for consistent style.
- Add `STPAlgorithmDefine.h` and move contents from `STPCoreDefine.h`.
- Suffix template name with `.in` to avoid confusion.
- Rename `STPRainDropSetting.hpp` to `STPRainDropSetting.h` to avoid confusion.
- Pre-compile `glad.c` and link to avoid duplicate complications.
- Further reduces compile time by moving definitions to source files instead of inlining.
- Remove some redundant CMake codes.