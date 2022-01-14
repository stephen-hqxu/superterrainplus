target_sources(${STP_REALISM_ENGINE} PRIVATE
STPRealismDefine.h
STPScenePipeline.h
STPScenePipeline.inl
# /Environment
Environment/STPAtmosphereSetting.h
Environment/STPCameraSetting.h
Environment/STPMeshSetting.h
Environment/STPOrthographicCameraSetting.h
Environment/STPPerspectiveCameraSetting.h
Environment/STPSunSetting.h
# /Object
Object/STPBuffer.h
Object/STPFrameBuffer.h
Object/STPImageParameter.hpp
Object/STPPipelineManager.h
Object/STPProgramManager.h
Object/STPProgramManager.inl
Object/STPRenderBuffer.h
Object/STPSampler.h
Object/STPShaderManager.h
Object/STPShaderManager.inl
Object/STPTexture.h
Object/STPVertexArray.h
# /Renderer
Renderer/STPCascadedShadowMap.h
Renderer/STPHeightfieldTerrain.h
Renderer/STPLightSpectrum.h
Renderer/STPPostProcess.h
Renderer/STPSun.h
# /Utility/Camera
Utility/Camera/STPCamera.h
Utility/Camera/STPOrthographicCamera.h
Utility/Camera/STPPerspectiveCamera.h
# /Utility
Utility/STPDebugCallback.h
Utility/STPIndirectCommand.hpp
Utility/STPLogStorage.hpp
Utility/STPRandomTextureGenerator.cuh
)

# config file for the realism rendering engine
# output the path to the GLSL shader code
set(SuperRealism+ShaderPath ${SuperRealism+Include}/Shader)
setupConfigTemplate(${STP_REALISM_ENGINE} STPRealismInfo.h.in Include/STPRealismInfo.h STPRealismInfo.h)