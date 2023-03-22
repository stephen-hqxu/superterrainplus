target_sources(${STP_REALISM_ENGINE} PRIVATE
STPExtendedScenePipeline.h
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
Environment/STPStarfieldSetting.h
Environment/STPSunSetting.h
Environment/STPTessellationSetting.h
Environment/STPWaterSetting.h
# /Geometry
Geometry/STPPlaneGeometry.h
# /Object
Object/STPBindlessTexture.h
Object/STPBuffer.h
Object/STPFrameBuffer.h
Object/STPGLVector.hpp
Object/STPImageParameter.hpp
Object/STPNullableObject.hpp
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
Scene/STPExtendedSceneObject.hpp
Scene/STPMaterialLibrary.h
Scene/STPSceneLight.h
Scene/STPSceneObject.hpp
# /Utility
Utility/STPAsyncAccelBuilder.h
Utility/STPCamera.h
Utility/STPDebugCallback.h
Utility/STPIndirectCommand.hpp
Utility/STPLogHandler.hpp
Utility/STPRandomTextureGenerator.cuh
Utility/STPRendererErrorHandler.hpp
)

set(STP_REALISM_ENGINE_SHADER
# /Shader
Shader/STPAlphaCulling.frag
Shader/STPAmbientOcclusion.frag
Shader/STPAurora.frag
Shader/STPBidirectionalScattering.frag
Shader/STPDeferredShading.frag
Shader/STPGaussianFilterKernel.frag
Shader/STPHeightfieldTerrain.frag
Shader/STPHeightfieldTerrain.geom
Shader/STPHeightfieldTerrain.tesc
Shader/STPHeightfieldTerrain.tese
Shader/STPHeightfieldTerrain.vert
Shader/STPPlaneGeometry.comp
Shader/STPPostProcess.frag
Shader/STPScreen.vert
Shader/STPShadowDepthPass.frag
Shader/STPSkybox.vert
Shader/STPStarfield.frag
Shader/STPSun.frag
Shader/STPSunSpectrum.comp
Shader/STPWater.frag
# /Shader/Common
Shader/Common/STPAnimatedWave.glsl
Shader/Common/STPAtmosphericScattering.glsl
Shader/Common/STPCameraInformation.glsl
Shader/Common/STPGeometryBufferWriter.glsl
Shader/Common/STPLightSpaceInformation.glsl
Shader/Common/STPMaterialRegistry.glsl
Shader/Common/STPNullPointer.glsl
Shader/Common/STPRayTracedIntersectionData.glsl
Shader/Common/STPSeparableShaderPredefine.glsl
# /Shader/RayTracing
Shader/RayTracing/STPFragmentUtility.cuh
Shader/RayTracing/STPGeometryAttributeFormat.hpp
Shader/RayTracing/STPInstanceIDCoder.cuh
Shader/RayTracing/STPScreenSpaceRayIntersection.cu
Shader/RayTracing/STPScreenSpaceRayIntersection.cuh
Shader/RayTracing/STPVectorUtility.cuh
)

target_sources(${STP_REALISM_ENGINE} PRIVATE
${STP_REALISM_ENGINE_SHADER}
)

# don't bother building them, they are all runtime compiled
set_source_files_properties(${STP_REALISM_ENGINE_SHADER}
TARGET_DIRECTORY ${STP_REALISM_ENGINE}
PROPERTIES HEADER_FILE_ONLY TRUE
)