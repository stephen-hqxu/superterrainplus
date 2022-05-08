set(STP_REALISM_OBJECT
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
)
set(STP_REALISM_SCENE
# /Scene
Scene/STPAlphaCulling.cpp
Scene/STPAmbientOcclusion.cpp
Scene/STPBidirectionalScattering.cpp
Scene/STPGaussianFilter.cpp
Scene/STPHeightfieldTerrain.cpp
Scene/STPPlaneGeometry.cpp
Scene/STPPostProcess.cpp
Scene/STPScenePipeline.cpp
Scene/STPScreen.cpp
Scene/STPSun.cpp
Scene/STPWater.cpp
)
set(STP_REALISM_SOURCE
STPCamera.cpp
STPCascadedShadowMap.cpp
STPDebugCallback.cpp
STPLightShadow.cpp
STPLightSpectrum.cpp
STPMaterialLibrary.cpp
STPOrthographicCamera.cpp
STPPerspectiveCamera.cpp
STPRealismEnvironment.cpp
STPRendererInitialiser.cpp
STPSceneLight.cpp
STPSceneObject.cpp

STPRandomTextureGenerator.cu
)

#shading and rendering library
add_library(${STP_REALISM_ENGINE} SHARED
${STP_REALISM_OBJECT}
${STP_REALISM_SCENE}
${STP_REALISM_SOURCE}

${STP_REALISM_ENGINE_INTERFACE}
)

# add source file filters
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}
PREFIX Source
FILES ${STP_REALISM_OBJECT} ${STP_REALISM_SCENE} ${STP_REALISM_SOURCE}
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