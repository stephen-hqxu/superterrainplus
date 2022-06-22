set(STP_REALISM_ENGINE_INTERFACE
STPRealismDefine.h
STPRendererInitialiser.h
STPScenePipeline.h
# /Environment
Environment/STPAtmosphereSetting.h
Environment/STPAuroraSetting.h
Environment/STPBidirectionalScatteringSetting.h
Environment/STPCameraSetting.h
Environment/STPLightSetting.h
Environment/STPMeshSetting.h
Environment/STPOcclusionKernelSetting.h
Environment/STPOrthographicCameraSetting.h
Environment/STPPerspectiveCameraSetting.h
Environment/STPStarfieldSetting.h
Environment/STPSunSetting.h
Environment/STPTessellationSetting.h
Environment/STPWaterSetting.h
# /Geometry
Geometry/STPPlaneGeometry.h
# /Object
Object/STPBindlessBuffer.h
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
# /Scene/Component
Scene/Component/STPAlphaCulling.h
Scene/Component/STPAmbientOcclusion.h
Scene/Component/STPAurora.h
Scene/Component/STPBidirectionalScattering.h
Scene/Component/STPGaussianFilter.h
Scene/Component/STPHeightfieldTerrain.h
Scene/Component/STPPostProcess.h
Scene/Component/STPScreen.h
Scene/Component/STPSkybox.h
Scene/Component/STPStarfield.h
Scene/Component/STPSun.h
Scene/Component/STPWater.h
# /Scene/Light
Scene/Light/STPAmbientLight.h
Scene/Light/STPCascadedShadowMap.h
Scene/Light/STPDirectionalLight.h
Scene/Light/STPLightShadow.h
Scene/Light/STPLightSpectrum.h
Scene/Light/STPShadowMapFilter.hpp
# /Scene
Scene/STPMaterialLibrary.h
Scene/STPSceneLight.h
Scene/STPSceneObject.h
# /Utility/Camera
Utility/Camera/STPCamera.h
Utility/Camera/STPOrthographicCamera.h
Utility/Camera/STPPerspectiveCamera.h
# /Utility
Utility/STPDebugCallback.h
Utility/STPIndirectCommand.hpp
Utility/STPLogHandler.hpp
Utility/STPRandomTextureGenerator.cuh
)

list(TRANSFORM STP_REALISM_ENGINE_INTERFACE PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set(STP_REALISM_ENGINE_INTERFACE ${STP_REALISM_ENGINE_INTERFACE} PARENT_SCOPE)

# config file for the realism rendering engine
# output the path to the GLSL shader code
set(SuperRealism+ShaderPath ${SuperRealism+Include}/Shader)
setupConfigTemplate(${STP_REALISM_ENGINE} STPRealismInfo.h.in Include/STPRealismInfo.h STPRealismInfo.h)