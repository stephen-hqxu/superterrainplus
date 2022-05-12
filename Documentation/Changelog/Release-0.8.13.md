# Release 0.8.13 - Improvement to Texture System

## SuperTest+

- Apply unit and coverage testing to `STPTextureDefinitionLanguage`.
- Apply unit testing to new functions in `STPChunk`.

## STPWorldPipeline

- Major optimisation to `STPWorldPipeline`. It now caches rendering buffer when the central chunk changes, and only load chunk map from `STPChunkStorage` and compute splatmap when old chunk map is not available to be reused.
- Type of rendering buffer and cache are all now identified by indices instead of some variable names. This greatly reduced the coding effort such that to update buffers for every type only a single `for` loop is required.
- Refactor some functions to cut down line-of-code.

## General fixes and improvement

- Removed function `getChunkCoordinate()` in `STPChunk` as it is not being used.
- Add a `UnuseTextureType` identifier in `STPTextureFactory` for indicating the value when texture type is not used in the texture type dictionary.
  - Also passes this new identifier to the shader and check for unused texture type before searching for texture data in the array.