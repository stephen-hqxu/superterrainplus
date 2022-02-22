#shading and rendering library
add_library(${STP_REALISM_ENGINE} SHARED
STPAlphaCulling.cpp
STPAmbientOcclusion.cpp
STPBindlessBuffer.cpp
STPBindlessTexture.cpp
STPBuffer.cpp
STPCamera.cpp
STPCascadedShadowMap.cpp
STPDebugCallback.cpp
STPFrameBuffer.cpp
STPGaussianFilter.cpp
STPHeightfieldTerrain.cpp
STPLightSpectrum.cpp
STPOrthographicCamera.cpp
STPPerspectiveCamera.cpp
STPPipelineManager.cpp
STPPlaneGeometry.cpp
STPPostProcess.cpp
STPProgramManager.cpp
STPRealismEnvironment.cpp
STPRenderBuffer.cpp
STPRendererInitialiser.cpp
STPSampler.cpp
STPSceneLight.cpp
STPSceneObject.cpp
STPScenePipeline.cpp
STPScreen.cpp
STPShaderManager.cpp
STPSun.cpp
STPTexture.cpp
STPVertexArray.cpp

STPRandomTextureGenerator.cu
)
set_target_properties(${STP_REALISM_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERREALISMPLUS_EXPORTS)

target_compile_definitions(${STP_REALISM_ENGINE}
PRIVATE STP_REALISM_IMPLEMENTATION
)