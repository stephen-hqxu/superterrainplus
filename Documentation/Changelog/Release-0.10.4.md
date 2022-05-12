# Release 0.10.4 - Real-time Shadow

## Terrain shadow

- Add `STPOrthographicCamera` as an orthographic projection camera, along with `STPOrthographicCameraSetting` to store settings.
- Add another type of flag to determine which component will contribute to shadow map to `traverse()` function in `STPScenePipeline`.
- Add `STPBindlessTexture` for handling automatic unresident of bindless texture handle.

### STPCascadedShadowMap

This is a shadow manager that implements CSM technique. It allows rendering shadow on a large scene using directional light with adaptive shadow quality based on distance from camera to the main scene.

`STPSun` now derives from this class, `STPHeightfieldTerrain` has provided users with two options, to cast or not cast shadow.

## General fixes and improvement

- Change exception type thrown when validating `STPSunSetting` and `STPAtmosphereSetting` in `STPSun` to `STPInvalidEnvironment`.
- Refactor *Lottes* and *Uncharted2* tone mapping calculation.
- Move `Near` and `Far` setting from `STPPerspectiveCameraSetting` to `STPCameraSetting` as these settings are shaded by both types of projection.
- Make all functions in `STPImageParameter` public.
- Add a new pure virtual function in `STPCamera` that allows overriding the near and far plane to calculate the projection matrix.
- Add `compareFunction()` and `compareMode` to `STPImageParameter` as well as implementations for all derived classes.
- Add `drawBuffer()` and `readBuffer()` for `STPFrameBuffer`.
- Fix incorrectly calculated tangent and bi-tangent for terrain unit plane that causes the normal Y direction to be flipped. Also change the unit plane winding order from CW to CCW for consistency reason.
- Fix a repetitive use of RNG in `STPLayerCache` test program that causes warning due to generation of the same RNG sequence to be generated during test session.
- Change the specification of all cameras. Now they all use callback functions to determine if the camera state has changed rather than having them returned from the camera matrix retrieval call.
- Instead of using an exponential function, terrain LoD shifting now uses linear. Remove `ShiftFactor` setting from `STPMeshSetting`.
- Buffer stream in `STPWorldPipeline` is now initialised as non blocking.
- `supply()` function in `STPBiomeFactory` now returns an instance rather than a pointer.
- Remove redundant storage of list iterator in `STPSingleHistogramFilter`, now the queue returns a moved memory block directly.

### Shader compilation

- Add some wrappers function over shader source code in `STPShaderManager` that allows pre-processing the source string directly rather than messing up other functionalities with the shader class.
- Remove all unnecessary shader include paths from all renderers (`STPSun` etc.) because all shader source code use absolute path.
- Fix incorrect semantics in `STPShaderManager`, shader include path should be an relative directory rather than the full name of the include source.
- `STPShaderManager` now requires explicit initialisation for system shader include. For such reason, added `STPRendererInitialiser` dedicated for initialisations.