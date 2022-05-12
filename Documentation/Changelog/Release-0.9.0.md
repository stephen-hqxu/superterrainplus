# Release 0.9.0 - Photorealistic Rendering Engine

*Please check out #31 for more development notes* :+1:

## SuperRealism+

This is the newly introduced rendering engine dedicated for providing photorealistic experience. All rendering tasks previously exists in the demo program have all been moved here.

It provides simple utilities for managing GL objects such as shader and buffer in a RAII manner. Also provides helper functions to manipulate such objects in an object-oriented way.

The realism engine provides built-in support for debug callback and shading language include, also allows usage in a multi-context environment.

Other utilities such as camera and GL error handling system are also be found.

### STPSun

A photorealistic sun and sky rendering. Sun is simulated like how we can observe the Sun on the Earth with day-night cycle and physical effects. Sky is rendered based on the location of the sun with atmospheric scattering using two-phase sphere-ray marching.

### STPHeightfieldTerrain

Aims to replace the old `STPProcedural2DINF` located in the demo program. Currently it functions in the exact same way as it predecessor. As the purpose of the demo program is to provide a minimal working example for users who wish to use our library, doing this reduces user efforts to implement shading code.

## SuperDemo+

In this release, we have removed all renderers from the demo program and restructure `STPStart` to improve general style and programming practice. In addition, as `SuperRealism+` is now a self-contained rendering engine, we decided to abandon `SglToolkit`.

## General fixes and improvement

- Add `STPFile` as a simple file manipulator that reads all lines from a file. It also provides simple utilities for processing filenames.
- Further reduced the amount of memory transaction between host and device in `STPWorldPipeline`.
- CMake target include and link refactoring. remove redundant codes.
- CMake configuration file outputs are now concentrated in a single directory with different names rather than assigning different filenames based on configuration.
- Add a new exception `STPGLError`.
- Move `STPMeshSetting` to the realism engine. Remove all getters from `STPConfiguration`.
- Add fractal noise function in `STPSimplexNoise`.