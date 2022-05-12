# Release 0.9.0-Preview.3 - Setup Scene Pipeline

## SuperRealism+

- Remove `STPContextStateManager` as it is being redundant and unhelpful.
- Add `STPScenePipeline` as a master rendering pipeline for the entire procedural terrain.
- Setup `STPHeightfieldTerrain` as a heightfield-based terrain renderer. It aims to replace `STPProcedural2DINF` in the demo program.
- Improve `STPShaderManager`, add a helpful function to allow to define macros in the shader source before compilation.
- To reduce verbosity, remove all error handling in `STPDebugCallback` except important functions.
- Also implement a shader include internal cache in `STPShaderManager` for auto shader include and cross GL context support.
- Fix incorrect shader include directory.
- Fix illegal memory access in `STPHeightfieldTerrain` where reading source code as a reference from a temporarily created `STPFile`.
- Fix incorrect condition check during shader include and keeps throwing exception for not found include source.
- Fix issues in the camera class for mixing up degrees and radians.
- Move camera setting structures to separate files under `STPEnvironment` namespace.

## SuperDemo+

- Remove `SglToolkit` as dependencies, the demo program is now self-contained as a rendering engine.
- Remove all renderers because `SuperRealism+` will do the job instead.
- Restructure the starter function `STPStart`.
- Improve general coding practice and readability.

## General fixes and improvement

- Fix the extension directive for GLSL shader, prefixed with `GL_`.
- Remove state record in `STPDebugCallback` and instead using GL query functions. This allows better support to multiple GL contexts.
- Add a simple helper function in `STPFile` for build filename.
- Replace shader filename processing in `STPSun` with compile-time functions instead of using `std::string` to allocate memory.
- Remove deprecated notes in the Readme, add more references.
- Change returned value in `STPTextureFactory`, instead of throwing exception when type is unregistered, it returns a flag.
- Remove all getters from `STPConfiguration` as it is very unnecessary.
- Fixed an issue in `STPShaderIncludeManager` that incorrectly deletes an array with scalar deleter.
- Move `STPMeshSetting` to `SuperRealism+`.