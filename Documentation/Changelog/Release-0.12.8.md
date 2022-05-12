# Release 0.12.8 - Ambient Occlusion

We introduced SSAO in release [0.11.8](https://github.com/stephen-hqxu/superterrainplus/releases/tag/v0.11.8) as a fundamental technique for ambient occlusion. In this release a more modern ambient occlusion technique is brought to *SuperTerrain+*.

## Horizon-based ambient occlusion

- Implement HBAO based on the current implementation of ambient occlusion.
- Enable switching between different AO algorithms in `STPAmbientOcclusion`.
- Rename `STPScreenSpaceAmbientOcclusion.glsl` to `STPAmbienOcclusion.glsl` for generality.
- Refactor and optimise view space geometry calculation.

## General fixes and improvement

- Explicit initialisation to `SQLite` database during `SuperTerrain+` engine initialisation.
- Fix a undefined behaviour when naming the texture database `STPTextureDatabase` where `operator+` is used between a `const char*` and `unsigned int`. Surround the integer with `to_string` to fix the issue.
- Add `STPProjectionCategory` in `STPCamera` to identify the type of projection of a camera instance.
- Use `offsetof` operator to identify variable offset instead of using hardcoded offset in `STPScenePipeline::STPCameraInformationMemory`.
- Add support for GL extension *GL_NV_representative_fragment_test*.
- Refactor heightfield triple texture scale blending system to cut down the scale of branching and divergence.
- Deprecate and remove support for logarithmic depth buffer because it is not friendly to depth reconstruction. We will consider using reversed depth buffer in the future release.
- Add ambient occlusion texture for the demo application.