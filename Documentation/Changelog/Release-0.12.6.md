# Release 0.12.6 - Simplify Lighting System

The major focus of this release is to replace the previously complicated uniform light list with simple pointers to buffer. A pointer can be used instead of having all different types of arrays and locate light and shadow information using indices.

## Shadow

`STPLightShadow` receives some improvement. Now `STPScenePipeline` no longer manges shadow memory and instead the light shadow class holds depth texture and framebuffer, and more importantly, a buffer object where a buffer address can be extracted and send to the shader.

The pointer approach significantly reduces the amount of data shared between light shadow instance and the scene pipeline because most shadow data can be sent directly to GPU via the address.

## Light

`STPSceneLight` receives a major overhaul. Different types of lights are broken down as children of `STPSceneLight`. We removed `STPEnvironmentLight` and replaced it with two new types of lights, `STPAmbientLight` and `STPDirectionalLight`. In addition, `STPSun` is no longer a child of any light source but `STPEnvironmentObject`, which is a new type of scene opaque object.

Likewise, light data are now shared with the shader using buffer address. This significantly reduces the complexity of our application. Now all light settings are associated with each light and it is no longer necessary to set light properties such as light strength and light direction via the scene pipeline.

## General fixes and improvement

- Fix various spelling errors in the source code documentation.
- Each shadow-casting light now initialises a shadow by moving a `unique_ptr` enclosed shadow instance rather than taking many constructor arguments.
- Add a new function for `STPBuffer` which allows copying sub-data from another `STPBuffer` instance.
- `STPLightSpectrum` is now a concrete class, all functionalities from all previous children are merged to the base class.
- Merge `STPSun::STPSunSpectrum` with `STPSun`.
- Replace texture lookup function `texture()` with `textureLod(..., 0)` for texture that are not intended to have level-of-details in shaders.

### Renderer logging system

- Remove log storage instances from all rendering components.
- Replace `STPLogStorage` with `STPLogHandle` which allows user to provide a pointer to implementation which defines how a log string should be handled.
- `STPShaderManager`, `STPProgramManager` and `STPPipelineManager` no longer returns a string of log. Instead all logs will be directed to the pointer provided by `STPLogHandle::ActiveLogHandler`.