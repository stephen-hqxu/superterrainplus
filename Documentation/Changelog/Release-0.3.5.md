# Release 0.3.5 - Simplification

## Highlights

There're not many major functional changes introduced in this release, but fixes and simplification to our program so less *setup* work is required before launching the terrain generator.

- Remove `STPHeightfieldSettings` from CUDA constant memory due to some performance concern about random access, especially during the phase of hydraulic erosion. Now `STPHeightfieldSettings` is initialised stateful, bounded to `STPHeightfieldGenerator` class.
- Remove the necessity to call `setErosionIterationCUDA()` separately, erosion iteration count is now stored in `STPRainDropSettings` and terrain generator will initialise erosion generator automatically during object construction.
- Cache `ErosionBrushIndices` and `ErosionBrushWeights` in `STPRainDropSettings` on host memory using `std::vector` so the code base is more neat without the need to implement all copy/move constructor/assignment operator explicitly.
- Add function `makeDeviceAvailable()` and `omitDeviceAvailable()` in `STPRainDropSettings` to give user control when to transfer host cache to device memory and invalidate device memory.
- Separate header and source code for `STPRainDropSettings` to cut down compile time.

## Fixes and improvement

- Move the definition of some private class/structure to source file so they are opaque.
- Fix incorrectly set chunk rendering buffer loading checker due to the removal of some previously deprecated features, causing all rendering chunks to be mapped and reloaded every frame even if all of them have been loaded.