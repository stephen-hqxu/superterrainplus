set(STP_REALISM_ENGINE_SOURCE
STPCamera.cpp
STPDebugCallback.cpp
STPOrthographicCamera.cpp
STPPerspectiveCamera.cpp
STPPlaneGeometry.cpp
STPRealismEnvironment.cpp
STPRendererInitialiser.cpp
STPScenePipeline.cpp

STPRandomTextureGenerator.cu
# /Object
Object/STPBindlessBuffer.cpp
Object/STPBindlessTexture.cpp
Object/STPBuffer.cpp
Object/STPFrameBuffer.cpp
Object/STPPipelineManager.cpp
Object/STPProgramManager.cpp
Object/STPRenderBuffer.cpp
Object/STPSampler.cpp
Object/STPShaderManager.cpp
Object/STPTexture.cpp
Object/STPVertexArray.cpp
# /Scene/Component
Scene/Component/STPAlphaCulling.cpp
Scene/Component/STPAmbientOcclusion.cpp
Scene/Component/STPAurora.cpp
Scene/Component/STPBidirectionalScattering.cpp
Scene/Component/STPGaussianFilter.cpp
Scene/Component/STPHeightfieldTerrain.cpp
Scene/Component/STPPostProcess.cpp
Scene/Component/STPScreen.cpp
Scene/Component/STPSkybox.cpp
Scene/Component/STPStarfield.cpp
Scene/Component/STPSun.cpp
Scene/Component/STPWater.cpp
# /Scene
Scene/STPCascadedShadowMap.cpp
Scene/STPLightShadow.cpp
Scene/STPLightSpectrum.cpp
Scene/STPMaterialLibrary.cpp
Scene/STPSceneLight.cpp
Scene/STPSceneObject.cpp
)

#shading and rendering library
add_library(${STP_REALISM_ENGINE} SHARED
${STP_REALISM_ENGINE_SOURCE}
${STP_REALISM_ENGINE_INTERFACE}
)
# add source file filters
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}
PREFIX Source
FILES ${STP_REALISM_ENGINE_SOURCE}
)
# add header file filters
source_group(
TREE ${STP_MODULE_INCLUDE}
FILES ${STP_REALISM_ENGINE_INTERFACE}
)

set_target_properties(${STP_REALISM_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERREALISMPLUS_EXPORTS)

target_compile_definitions(${STP_REALISM_ENGINE}
PRIVATE STP_REALISM_IMPLEMENTATION
)