# Release 0.4.0-Preview.3 - Generalised Heightmap Generator

In this release we aim to separate heightmap generator from `STPHeightfieldGenerator` and allow user and developer to create their own algorithms and parameter sets for each biome. Scripts for heightmap generator will be compiled in runtime.

## Highlights

- Add a high-level wrapper class `STPDiversityGenerator` which includes high-level functions for easy runtime-compiled CUDA scripts.
- Add various helper functions in the class above for script complication, linking and management.
- Extend `STPDeviceErrorHandler` so it can handle all APIs used (cuda, cudart and nvrtc) within our program.

> NVRTC is required in order to compile and execute runtime script.

## General fixes and improvement

- Fix some documentation typos.
- Move some header includes into source file to reduce compile time, and remove unnecessary headers.
- Fix incorrectly defined copy/move constructor/assignment operator in various classes.
- Add `virtual` to destructor of classes containing pure virtual function(s).