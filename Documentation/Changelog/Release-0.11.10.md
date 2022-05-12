# Release 0.11.10 - Aerial Perspective

## Horizon extinction effect

To avoid having the edge of the terrain getting cut out with a very sharp border as it approaches camera far clipping plane, we proposed a fix by blending the atmosphere with the extinction area on the terrain.

This effect is achieved by using alpha blending on the extinction area with the sky. The alpha channel on the geometry is determined by the distance from fragment to camera.

Instead of perform a full-screen ray marching to do the sky blending, `STPAlphaCulling` is implemented to modify stencil buffer such that only geometries in the extinction region are included in the environment rendering.

The extinction parameter can be adjusted in `STPScenePipeline`.

## General fixes and improvement

- `STPPostProcess` now supports alpha channel framebuffer.
- Fixed the alpha channel of the default clear colour in scene pipeline to 1.0, was 0.0 previously.
- The program now uses real timer to calculate, for example sun position, rather than using frametime. Such that the time step is independent of FPS.
- Declare all static-declared lambda expression variables as `constexpr` as well, in the entire project .
- Make the destructor of `STPLocalRNG` in `STPLayer` as `default` in the header.
- Adjusted chunk scale to make sure the camera cannot see the border of rendered chunks.
- Making slight adjustment to the shadow bias to reduce shadow acne flickering.
- Enable robust context when the demo program is compiled in debug mode.
- Improve permutation table generation in `STPPermutationGenerator`. Replace raw array with `std::array` and initial table is generated in compile time instead of hand-typed. Rename inappropriate variable name.