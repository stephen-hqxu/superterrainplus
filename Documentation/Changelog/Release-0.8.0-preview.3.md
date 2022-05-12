# Release 0.8.0-Preview.3 - Setup Texture Splatmap Generator

## Texture utilities

We plan to split texture splating into two stages: splatmap generation which will be done with CUDA before rendering, and terrain texture splatting using the splatmap generated in GLSL shader.

### STPTextureFactory

- Refactor to make code more readable and maintainable.
- Cut down the class size by moving most standard library containers inside the function as local variable.

### STPTextureDatabase

- Allow accessing splat rule data with `getSplatDatabase()`.
- Rename some of the texture semantics:
  - TextureCollection -> Texture : contains multiple maps of different types.
  - TextureData -> Map : individual texture data of a specific type.
- Add some remove functions to allow dropping texture from the database.
- Allow `addTexture()` function to take a parameter `count` to insert `count` number of texture at once, and return an array of all texture IDs.
- `STPTextureDatabase` and `STPTextureSplatBuilder` are now move-constructable and move-assignable.
- Improve code usability for batch queries.

## Texture utility testing

- Add `STPTestTexture` to `SuperTest+`.
- Fix potential exception throw when prepared statement `sqlite3_stmt` is finalised, since `finalize()` may return error codes from the previous operation. This has been mitigated by resetting prepare statement if error is detected.
- Fix syntactic errors in the templates `addAltitudes()` and `addGradients()` that cause failed compilation.
- Fixing incorrect use of blob data type.

## General improvement and fixes

- Update the concurrency calculator in `STPChunkProvider` so it matches the latest free-slip algorithm.
- Provide solutions for #25 and #26.
- Refactor `STPVoronoiLayer` and move namespace alias declaration inside the function.
- Separate the runtime compiler from `STPDiversityGeneratorRTC`, and now it's presented as a stand-alone function `STPRuntimeCompilable`.
- Remove redundant multithreading in `STPMasterRenderer`.

> In order to use runtime compilation feature for `STPDiversityGenerator`, one must inherit both `STPDiversityGenerator` and `STPRuntimeCompilable`, a.k.a., multiple inheritance.

- Link OpenGL as an interface target with our custom `GLAD` build target in CMake, because GLAD is usually used in conjunction with OpenGL, it doesn't make sense to take them apart.
- Move `STPSQLite` to the `SuperTerrain+` root directory.
- Add a OpenGL compatibility header `STPOpenGL` and replace all GL type presence exposed in headers with such. This can be used to reduce unnecessary library include and linking.
- Improvement to `README`.
- Update `LICENSE`.