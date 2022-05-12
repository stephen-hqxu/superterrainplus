# Release 0.9.0-Preview.1 - Rendering Toolkit

*Note: You may need to recompile the entire application for this release.*

## SuperRealism+

We introduce a new build target (or library) dedicated for shading and rendering.
- `STPShaderManager` is a smart GLSL shader object manager that takes source code and compile it with error handling.
- `STPProgramManager` is a GLSL program object manager that takes some shader managers with link and validation error handling.
- `STPPipelineManager` is a GLSL program pipeline object manager that recombines program objects into a new program.

### STPSun

We introduce `STPSun` as an utility for global light source on the procedural terrain, with `STPSunSetting` for storing the settings.

`STPSun` is a physically-based sun position calculator that simulates how the Sun rotates around the Earth in real life with seasonal effects.

## General fixes and improvement

- Reduce the number of memory operations further in `STPWorldPipeline`. Now the front rendering buffer is only cleared as per-chunk basis when there is no old buffer to be used instead of clearing the entire buffer at the beginning.
  - Also clear buffer is now a pitched device memory rather than a pinned host memory.
- Texture anisotropy levels can now be adjusted via `Engine.ini` instead of being hard-coded in `STPWorldManager`.
- Add a fractal utility in `STPSimplexNoise` for fast fractal simplex noise generation.
- Fix an issue in algorithm device library that causes warning during compilation due to use of non-`const` in a constant function.
- Simplify CMake library include and link structure, remove redundant codes.
- Rename shared library exports macros for consistency.
- Introduce `STPNullablePrimitive` in the main engine to allow using a primitive type in `std::unique_ptr` without allocating dynamic memory.
- Move terrain normalmap generation to fragment shader for higher level-of-detail and accuracy.
- Terrain splatmap generation now uses fractal simplex noise instead of a single one.