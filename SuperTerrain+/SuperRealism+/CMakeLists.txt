#shading and rendering library
add_library(${STP_REALISM_ENGINE} SHARED
STPBuffer.cpp
STPCamera.cpp
STPDebugCallback.cpp
STPHeightfieldTerrain.cpp
STPLightSpectrum.cpp
STPPerspectiveCamera.cpp
STPPipelineManager.cpp
STPProgramManager.cpp
STPRealismEnvironment.cpp
STPScenePipeline.cpp
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