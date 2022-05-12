# Release 0.12.0 - Multi-Scale Texturing

## Improve texturing system

### STPTextureDatabase

Introduce a new texture group, texture view group, which specifies the UV scale of a particular texture. Remember a texture is a collection of many texture maps, such that the view group settings will be applied to all texture maps under texture as well.

The new system allows specification of 3 different UV scales, and the scales will be chosen by the renderer based on user-specified distance to the texel.

- Remove multi-add functions such as `addTextures()` because they seem to be useless after the introduction of texture definition language.
- `addTexture()` now allows assigning, optionally, a user-specified name to the texture.
- Restructure texture group logic, rename *texture group* to *map group* to disambiguate.
- Refactor database record manipulation functions a little bit.

### STPTextureDefinitionLanguage

The specification of TDL has been updated to add support for the new texture view group system. The view group is specified using a new directive, `#group` and is identified by group type `view`.

For more information about the new syntax, check out TDL specification in the project documentation.

## General fixes and improvement

- Resolve issue #36.
- Simplify `STPSimplexNoise` calculation, remove redundant arithmetic operations.
- All camera-related calculations (viewer camera, light-space camera etc.) are now all done with `double` on host side before converting to `float` and sending to device.

### STPHeightfieldTerrain

- Fix an incorrect texture region dictionary boundary check logic in the fragment shader which could result in out-of-bound array access.
- Noise scale used for stratified sampling now uses `unsigned int` rather than `float`, because a real number will make the UV scale non-aligned with tile boundaries and the whole texture tiles move when switching centre chunk.
- Refactor texture smoothing using histogram bins instead of fetching texture in every iteration during splatmap smoothing to greatly reduce memory bandwidth overhead.
- Remove `UVScaleFactor` from `STPMeshSetting`, add `STPTextureScaleDistanceSetting` to control texture scale to be made active based on texel distance.