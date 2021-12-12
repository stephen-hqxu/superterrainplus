#shading and rendering library
add_library(${STP_REALISM_ENGINE} SHARED
STPBuffer.cpp
STPCamera.cpp
STPDebugCallback.cpp
STPHeightfieldTerrain.cpp
STPPerspectiveCamera.cpp
STPPipelineManager.cpp
STPProgramManager.cpp
STPRealismEnvironment.cpp
STPScenePipeline.cpp
STPShaderIncludeManager.cpp
STPShaderManager.cpp
STPSun.cpp
STPVertexArray.cpp
)
set_target_properties(${STP_REALISM_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERREALISMPLUS_EXPORTS)