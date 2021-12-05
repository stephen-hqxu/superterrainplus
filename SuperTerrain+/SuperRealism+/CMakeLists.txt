#shading and rendering library
add_library(${STP_REALISM_ENGINE} SHARED
STPPipelineManager.cpp
STPProgramManager.cpp
STPRealismEnvironment.cpp
STPShaderManager.cpp
STPSun.cpp
)
set_target_properties(${STP_REALISM_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERREALISMPLUS_EXPORTS)