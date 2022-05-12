# Release 0.11.5 - Smooth Shadow

## Shadow map filtering

- Implement the following shadow map filters:
  - Percentage-closer filtering
  - Variance shadow map
- Improve shadow biasing system. Add normal biasing and far plane bias scaling for directional shadow.
- Refactor shadow map filter selection in the scene pipeline. Different shadow map filters take different initialisation parameters.
- Add a new argument to `addDepthConfiguration` for opaque object, the new argument takes a pointer to depth fragment shader. This function will be called by the scene pipeline automatically.
- Introduce cross-cascade blending which resolves an ugly line when applying shadow filters at the cascade border. To resolve light bleeding introduced by the blending system, cascade plane can be biased when initialising `STPCascadedShadowMap`, this bias value is called *CascadeBandRadius*.
- Fine-tune some shadow settings in the demo program to make it looks better.

## General fixes and improvement

- Remove `virtual` declaration on the destructor on `STPCascadedShadowMap` because this class is no longer intended to be a base class.
- Simplify cascaded shadow map layer lookup process.
- Uniform locations of all light properties are now cached with `std::unordered_map` using light identifier and light type instead of performing string lookup every time.
- `STPProgramManager` now allows sending uniform data using uniform location.
- Add more overloads for `setLight()` in `STPScenePipeline`.
- Terrain renderer now allows using two different tessellation level of detail controls for normal rendering and depth rendering.
- Add support for logarithmic depth buffer, related settings can be found under `STPCameraSetting`.

> The log depth buffer currently does not work with shadow map because shadow calculations are not in log scale.

- Improve error handling for scene pipeline initialisation. Now all values set for the shadow parameters are checked for validity.
- Refactor `STPDeferredShading.glsl` to make it more organised.
- Force the shadow map to be a square texture. The resolution of all shadow maps is now specified by a single unsigned integer, namely extent length.

### Bug discovered

- Fix a bad sampling pattern when performing texture splatting smoothing during terrain rendering, which incorrectly assumes the range of coordinate causing most of the samples biased towards negative axis.
- Fix an issue where in the texture management system mip-map is used however only a single level of detail is allocated, causing no mip-map is actually being generated.
  - `STPTextureDatabase` now allows user to specify the number of mip-map level to allocate when creating a new texture group.