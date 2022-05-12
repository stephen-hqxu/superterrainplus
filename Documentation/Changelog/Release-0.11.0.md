# Release 0.11.0

We perform an overhaul to the master scene pipeline and all rendering components for deferred shading. Deferred shading makes designing an abstract engine much easier as light calculation can be concentrated into one shader instead of spreading around and distributing by shader include and macros. Also it is inevitable for more advanced lighting effect which will be added to the engine later such as SSAO, SSR and adding support for 1000+ lights.

## Deferred shading - Introduce Deferred Shading

- Add `STPGeometryBufferResolution` as a G-buffer manager and lighting processor. It current captures the following geometry data with data format of:
  - Albedo (RGB8)
  - Normal (RGB16_SNORM)
  - Specular (R8)
  - Ambient Occlusion (R8)
  - Depth Stencil (DEPTH32F_STENCIL8)

> Geometry position is reconstructed from depth buffer.

- Add a new common shader `STPGeometryBufferWriter` for writing geometry data into G-buffer from various rendering components.
- Rename `STPDirectionalLight` back to `STPCascadedShadowMap` to make it sounds more fancy.
- Remove all lighting calculation, light-related uniforms and constructor parameters from `STPHeightfieldTerrain`. Remove light settings from `STPMeshSetting`.
- For easy initialisation, sun spectrum is now a member of `STPSun`, so the sun acts as both a light source and a rendering component.

### Renderer abstraction

- Add `STPSceneObject` acting as a base abstract renderable objects. `STPHeightfieldTerrain` current inherits from it. This new base class also uses the shadow-casting logic and provide two different versions of objects.
- Add `STPSceneLight` basically for the same purpose as the scene object. There are many derivations of different types of light to be added in the future. Currently only `STPEnvironmentLight` is supported. `STPSun` now inherits from this derivation.
- Add `STPLightShadow` as a base class for all shadow implementations. `STPCascadedShadowMap` now inherits from this.
- Add `STPScreen` as a base off-screen renderer. The screen vertex data can be shared by various off-screen rendering targets such as `STPPostProcess`.
- Introduce the idea of depth renderer group that allows rendering opaque objects to depth texture with different light space size. The basic idea is by compiling GL shaders with all possible declared light space sizes.

### STPScenePipeline

The rendering pipeline has received the most amount of attention. It's now the master rendering controller and memory repository for deferred rendering, and shares memory buffer among rendering components.

To initialise a scene pipeline, one needs to use `STPScenePipelineInitialiser` to first add shadow-casting light to get shadow information, then the rests of the object. To reduce efforts for user to manage rendering component, all rendering components are held by the scene pipeline and user will get a pointer to the objects added.

`STPPostProcess` no longer hold its own texture and framebuffer, it now reads texture from arguments when `process()` function is called. This helps concentrating memory in the scene pipeline for the sake of deferred shading.
 
## General fixes and improvement

- Buffer resolution in `STPPostProcess` is checked for non-zero.
- For `STPTexture`, `Target` member variable is no longer declared as `const` to allow use of move assignment operator.
  - Regarding this modification, `STPPostProcess` now recreate screen buffer using move assignment rather than `std::optional`.
- Add a new function in `STPFrameBuffer` for specifying multiple framebuffer draw buffers as an array.
- The texture splatting system no longer supports displacement and emissive mapping. Add support for roughness mapping.
- Remove unused shader variables such as *position_clip* passed between each shader in terrain rendering pipeline.
- `STPLightSpectrum` now allows specifying the number of colour in the spectrum, as well as the channel format.
  - Now `STPStaticLightSpectrum` and `STPArrayLightSpectrum` all use monotonic colour (GL_TEXTURE_1D) with GL_RGB8 format.
  - `STPSunSpectrum` still remains the specification unchanged. using bi-tonic colour (GL_TEXTURE_1D_ARRAY) with GL_RGBA16F format.
- Declare `STPSetting` with virtual destructor. Remember to recompile all CUDA codes in *SuperTerrain+* module.
- Remove texture type *Displacement* and *Emissive*, add *Roughness*.