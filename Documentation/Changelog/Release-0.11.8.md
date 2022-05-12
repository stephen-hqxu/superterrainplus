# Release 0.11.8 - Ambient Occlusion

## Screen-space ambient occlusion

The focus in this release is the newly added `STPAmbientOcclusion` for performing SSAO which allows us to see the details on the terrain even when it is in shadow. The shader implementation can be found in `STPScreenSpaceAmbientOcclusion.glsl`. As always, the effect is configurable and the settings are located in `STPOcclusionKernelSetting`.

Also we introduced `STPGaussianFilter` which is a separable Gaussian blur kernel. It features a configurable Gaussian function. This class is currently used by the ambient occlusion calculator for blurring the final output.

## Improve off-screen rendering

We refactor `STPScreen` into two sub-classes, `STPScreenVertexShader` and `STPScreenVertexBuffer`. This allows sharing vertex shader during initialisation among all off-screen renderers and vertex buffer during rendering to safe memory and reduce number of binding operations.

Additionally each `STPScreen` base instance contains a `STPProgramManager` with some helper functions to simplify the process for setting up off-screen rendering. Please keep in mind that `drawScreen()` function no longer binds vertex buffer automatically, binding and program usage needs to be done by the user externally.

Also add `STPSimpleScreenFrameBuffer` under `STPScreen` as a refactor class for single colour attachment off-screen rendering.

## General fixes and improvement

- Add a texture barrier call in the main rendering loop before deferred light pass to ensure data safety since there might still be a potential feedback loop problem.
- Add extra `static_assert` for camera buffer structure to ensure the alignment is consistent with OpenGL `std430` alignment rule.
- For loading INI into the program, all setting names are made inlined to make it more readable. Add INI loader function for ambient occlusion settings.
- Simplify spectrum generation shader. Remove explicit uniform for spectrum size since it can be queried by calling `imageSize`.
- Change memory barrier function post spectrum generation from image access barrier to texture fetch barrier.
- Fix a bug in `STPSunSpectrum` for forgetting to assign a sampler binding to the shader. This did not lead to error as OpenGL assigns sampler binding 0 by default.
- `STPPostProcess` now holds its own memory rather than taking it from the scene pipeline.
- Add new function in `STPFrameBuffer` to allow detaching a target.
- All colour buffers are explicitly cleared instead of relying on the fact that they are all overdrawn to avoid running into bugs in the future.