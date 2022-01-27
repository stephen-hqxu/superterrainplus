target_sources(${STP_REALISM_ENGINE} PRIVATE
STPRealismDefine.h
STPRendererInitialiser.h
# /Environment
Environment/STPAtmosphereSetting.h
Environment/STPCameraSetting.h
Environment/STPLightSetting.h
Environment/STPMeshSetting.h
Environment/STPOrthographicCameraSetting.h
Environment/STPPerspectiveCameraSetting.h
Environment/STPSunSetting.h
# /Object
Object/STPBindlessTexture.h
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
Renderer/STPHeightfieldTerrain.h
Renderer/STPPostProcess.h
Renderer/STPScreen.h
Renderer/STPSun.h
# /Scene
Scene/STPCascadedShadowMap.h
Scene/STPLightSpectrum.h
Scene/STPSceneObject.hpp
Scene/STPScenePipeline.h
Scene/STPScenePipeline.inl
# /Utility/Camera
Utility/Camera/STPCamera.h
Utility/Camera/STPOrthographicCamera.h
Utility/Camera/STPPerspectiveCamera.h
# /Utility
Utility/STPDebugCallback.h
Utility/STPIndirectCommand.hpp
Utility/STPLogStorage.hpp
Utility/STPRandomTextureGenerator.cuh
Utility/STPShadowInformation.hpp
)

# config file for the realism rendering engine
# output the path to the GLSL shader code
set(SuperRealism+ShaderPath ${SuperRealism+Include}/Shader)
setupConfigTemplate(${STP_REALISM_ENGINE} STPRealismInfo.h.in Include/STPRealismInfo.h STPRealismInfo.h)