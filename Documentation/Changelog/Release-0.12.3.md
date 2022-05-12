# Release 0.12.3 - Optimisation to Renderer

## STPHeightfieldTerrain

- Addresses issue #35 by computing terrain mesh using double precision before casting to single precision and storing results.
- Introduce `STPPlaneGeometry` as a plane generator that allows specifying number of subdivision on a plane and optimised for fast and high-precision generation of high quality mesh.
- Refactor terrain shader to allow making use of the new plane generator and avoid precision problem.
- Simplify terrain texture splat region lookup using shader pointer. Remove registry dictionary from the terrain fragment shader, now region registry holds uses pointers to splat texture and null pointer to identify regions with no texture data.

## General fixes and improvement

- Introduce `STPBindlessBuffer` as a thin wrapper over buffer address. This requires extension *GL_NV_shader_buffer_load*, however.
- Remove extension check in `STPBindlessTexture` because this is mandatory anyway.
- Refactor academic report reference to a single file.
- Add a new GLSL common header `STPNullPointer.glsl` to help better interacting with use of pointers in shading language.
- `STPLightSpectrum` now manages its own texture handle instead of keeping an array of spectrum handle in `STPScenePipeline`.
- Add documentation for texture definition language.