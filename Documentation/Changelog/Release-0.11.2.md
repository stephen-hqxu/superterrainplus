# Release 0.11.2 - Dynamic Scene Pipeline

## Improve level of abstraction for STPRealism

This release makes `STPScenePipeline` dynamic, meaning all rendering components like opaque objects and lights, can be added to the scene pipeline after construction in runtime.

Due to the dynamic nature, all memory in the deferred shader needs to be declared prior to initialisation, and the number of objects added must not exceed this limit. This limit is set by the user. The limit will all be stored in `STPSceneShaderCapacity`.

- Include directory restructured. *Renderer* has been moved under *Scene* and renamed to *Component*. `STPScenepipeline` has been moved to the root.
- Shadow map texture is no longer shared because I don't find it necessary, each shadow-casting light is now given its own texture memory, such that the shadow map resolution can be adjusted on per-light basis. Each light is assigned with one framebuffer object.
- Add a new pure virtual function to `STPLightShadow` for getting shadow map resolution.
- Terrain renderer now allows adding shadow-casting light with different depth configuration dynamically, instead of passing `STPShadowInformation` in the constructor.
- Add `lightDirection()` virtual function in `STPEnvironmentLight<false>` so the scene pipeline can get the light direction automatically without asking inputs from the user.

## General fixes and improvement

- Address #34 that the demo program incorrectly assumes `glfwSwapBuffer()` will do implicit synchronisation (in fact it will not), which causes data racing.
- Fix a potential feedback loop (therefore undefined behaviour) for a bound depth buffer while reading it in the shader. It is fixed by only binding stencil attachment to the post process framebuffer as we don't need depth buffer for off-screen rendering.
- Make all shader storage buffer block named to avoid variable collision.
- Add result of projection * view matrix and its inverse to camera information buffer to further reduce the amount of runtime computation on GPU.
- Add a new member in light space information shader storage block to locate a light space matrix for a particular light.
- Rename `STPDepthRendererGroup` to `STPDepthRenderGroup` in `STPSceneObject`.
- Remove currently unused explicit instantiation for `STPRandomTextureGenerator` in `STPRealism`.
- With the new concept of depth render group, make light space shader storage buffer as a dynamic array and `STPShadowInformation` is no longer required because no information is required to shared with any opaque object.
- Remove log storage and status flags in `STPShaderManager`, `STPProgramManager` and `STPPipelineManager`. Also remove redundant validity check in various rendering components because those managers throw exception before validity check when there is error.
- `STPShaderManager` now allows setting a mandatory source name which will be prepended to the compilation log (if any) for easier identification of error. All rendering components now set this source name to the filename of shader.
- The shadow-casting version of terrain renderer now has an internal queue for holding up logs from compilation of depth renderer.
- `STPMemoryPool` in the main engine now uses `std::unordered_map` instead of sorted `std::vector` to search for memory chunk with given size because benchmark shows hash table is still faster than binary search even when the entry size is very small.
- Optimise view frustum corners calculate in `STPCascadedShadowMap`. Use SIMD for shadow frustum calculations.