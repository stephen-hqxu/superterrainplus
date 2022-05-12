# Release 0.9.3 - Procedural Sky Rendering

- Add a function in `STPTerrainParaLoader` to load sun and atmosphere settings from the INI.
- Fix some maths error in the sun shader.
- Fix more maths error in sun direction calculation function.
- Enable procedural sky rendering in the demo program. Enjoy the realism.
- Simplify calculation for scattering to improve performance.

## SuperRealism+

- Add `STPTexture` as a GL texture object manager.
- Add `STPRandomTextureGenerator` as a utility of noise texture generation for GLSL shader. It is done using *cuRAND* device library.
- Perform smoothing to texture splatmap using volumetric noise, now the edge of texture region blends more naturally to adjacent regions. Algorithm is built based on [Efficient Soft-Edged Shadows Using Pixel Shader Branching](https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using).

## General fixes and improvement

- Improve error handling in `STPStart` and catches exception thrown during main engine setup and rendering.
- Attach resource folder to the repo.
- Fix a typo *atmosphere* to *atmosphere*.
- Keep improving `STPWorldManager`, now it copies buffer from cache only when the chunk status has been marked as true previously. This avoids problem of chunk not getting refreshed when chunk is loaded buffer the central chunk switches.
- Update the camera rotation specification, instead of taking an absolute Cartesian position, it now takes a relative direction offset. This addresses the problem of having sudden camera rotation when the camera moved for the first time because the *lastOffsetPosition* was set to zero initially by default.
- Remove some types from `STPTextureType` that are probably never going to be used.
- Move most hard-coded settings in the shader to INI, following a suggestion in #32.
- Remove normal, tangent and bi-tangent attributes from heightfield renderer.