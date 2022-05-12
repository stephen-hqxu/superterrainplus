# Release 0.10.6 - Improvement to Shadow System

## Better shadow system

- Shadow map filter can now be chosen in runtime. It current supports nearest and bilinear filter. More shadow smoothing techniques will be implemented in the future.
- Rename `STPCascadedShadowMap` to `STPDirectionalLight` to make it more general.
- Move shader storage buffer for shadow mapping from `STPDirectionalLight` to `STPShadowPipeline` which is a new class.
- `STPSun` is now also given choices for casting or not casting shadows.
- Add `STPShadowInformation` for sharing shadow map setting from `STPShadowPipeline` to various renderers.
- `STPDirectionalLight` now uses camera callback to determine if light space matrices are outdated and only recompute them (and flush the buffer) whenever necessary.

With the new system, the correct order for setting up a rendering pipeline is:

- Create light that casts shadows.
- Create `STPShadowPipeline` with an array of pointers to light that just created.
- Create all the rest of the scene including light that does not cast shadow and opaque object.
- Create `STPScenePipeline` with a pointer to `STPShadowPipeline`.

As now lights hold memory owned by the scene pipeline, make sure lights are not writing to the buffer **after** the dependent scene pipeline is destroyed.

## General fixes and improvement

- Fix a potential undefined behaviour in terrain fragment shader for accessing array of bindless samplers in a non-uniform manner.
  - GL extension `NV_gpu_shader5` is now required in order to run the program.
- Remove unnecessary extension directives from some shader codes.
- Move bindless shadow handle from individual shader to shader storage buffer to be shared among all renderers.
- `STPCamera` now allows registering as many listeners as user wants.
- `STPScenePipeline` now deregisters itself from camera callback at destruction.
- Fix an incorrect return value semantics in shadow map sampling shader. The return value of that function represents the light intensity instead of shadow intensity, or *light intensity = 1.0 - shadow intensity*.
- Rename rendering function name from `operator()` to some more concrete name like `render()`.