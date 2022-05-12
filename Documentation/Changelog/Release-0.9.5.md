# Release 0.9.5 - Terrain Shading

## Basic lighting

- Fix a problem in terrain normalmap generation where normals are not normalised. Also fix the facing of the normalmap so that it is in terrain world space rather than tangent space.
- Add lighting to the terrain, light direction is based on the sun.
- Attach texture normalmap to the project. The engine currently supports:
  - Colormap, a.k.a., albedomap
  - Normalmap
  - Specularmap
  - Ambient-occlusionmap
- Implement 5 normalmap blending algorithms and they can be chosen during heightfield renderer initialisation.
- Move light settings to INI file.
- Improve terrain shader to handle all cases when each type of texture is not enabled.

## General fixes and improvement

- Improve level of abstraction for the scene pipeline and each rendering component. `STPScenePipeline` no longer changes component states, such as camera position, automatically before rendering. Internal buffers like shader storage buffers that are not exposed to the external will be managed automatically.
- Move renderer uniform update functions out of rendering loop. They are now updated explicitly using corresponding setters rather than doing it automatically during rendering loop. This helps reducing unnecessary GPU communication when not needed.
- Fix some hard-coded texture splatting region lookup in the shader.
- Add a function in world pipeline to retrieve the last central chunk position. Also made some changes to the return specification of the `load()` function, which now returns whether the chunk position has been changed.
- Further eliminate unnecessary GL uniform.
- Fix the terrain normal direction that causes incorrect lighting by flipping x and y components.
- Improve region searching algorithm to avoid repetitive texture look-up in the terrain fragment shader.
- Turned on level-4 compiler warning at debug mode. Also modify all source codes to ensure no warning is generated during compilation.