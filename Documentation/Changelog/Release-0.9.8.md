# Release 0.9.8 - Dynamic Lighting

## Terrain light colour

- Add 2 configurable light colour in the terrain shader:
  - Indirect: ambient
  - Direct: diffuse
- Allow changing individual lighting component colour.
- Introduce the idea of spectrum lighting that allows looking up light colour from a pre-computed light spectrum.
- Wrap the atmosphere scattering algorithm such that it can be used by both the sun renderer and spectrum emulator (which is a compute shader).

### STPLightSpectrum

A utility that allows storing terrain light colour(s) in an array for looking up in runtime. It currently provides:
- Static spectrum that defines a monotonic colour.
- Array spectrum that defines user-specified colours.
- Sun spectrum that uses `STPSun` to calculate the light spectrum based on the sun elevation.
- Enable sun spectrum lighting in the demo program.

## General fixes and improvement

- Allow loading displacement map in the demo program. Refactor texture loading and texture type checking.
- Fix an incorrect return type in the terrain shader.
- Improve sun direction calculation, remove member variable `LocalSolarTime` and change `Day` from unsigned long long to double. Now both time in a day and a year can be calculated using one variable.
- Replace some `size_t` with `unsigned long long` because the underlying type of `size_t` is platform-dependent.
- Change the specification of day-night cycle offset. Instead of using angle it now uses sun elevation vector.
- Add a function in `STPProgramManager` to allow to query work group size for compute shader program.
- Add shading language include extension check in `STPShaderManager` if any include path is specified.
- `STPTexture` now requires specifying which GL function to be called explicitly for texture storage and sub-image function.
- `finalise()` in `STPProgramManager` now returns pointer to log.
- Fix a bug in `STPProgramManager` that messes up shader type with shader reference ID.
- Remove the following settings from `STPAtmosphereSetting`:
  - LowerElevation
  - UpperElevation
  - CycleElevationOffset
- Improve exception handling in `STPLayerManager` to avoid memory leak on exception.
- Improve `STPLayerCache` caching mechanism. Instead of using a callback function, it now splits caching operation into 3 steps, locate address, read and write.
- Set `sample()` function to private in `STPLayer`.