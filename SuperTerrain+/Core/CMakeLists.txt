set(STP_MAIN_ENGINE_SOURCE
STPDeviceRuntimeBinary.cpp
STPDeviceRuntimeProgram.cpp
STPEngineInitialiser.cpp
STPEnvironment.cpp
STPException.cpp

STPHeightfieldKernel.cu
STPRainDrop.cu
# /Utility
Utility/STPFile.cpp
Utility/STPSmartDeviceObject.cpp
Utility/STPThreadPool.cpp
# /World/Diversity
World/Diversity/STPBiomeFactory.cpp
World/Diversity/STPLayer.cpp
World/Diversity/STPTextureDatabase.cpp
World/Diversity/STPTextureFactory.cpp
# /World
World/STPChunk.cpp
World/STPErosionBrushGenerator.cpp
World/STPFreeSlipTextureBuffer.cpp
World/STPHeightfieldGenerator.cpp
World/STPWorldPipeline.cpp
)

add_library(${STP_MAIN_ENGINE} SHARED
${STP_MAIN_ENGINE_SOURCE}
${STP_MAIN_ENGINE_INTERFACE}
)
# add source file filter
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}
PREFIX Source
FILES ${STP_MAIN_ENGINE_SOURCE}
)
# add header file filter
source_group(
TREE ${STP_CORE_INCLUDE}
FILES ${STP_MAIN_ENGINE_INTERFACE}
)

# set export preprocessor
set_target_properties(${STP_MAIN_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERTERRAINPLUS_EXPORTS)

# tell the compiler we are the source engine builder
# there are some preprocessors in the source code to warn or prevent external environment from including internal headers
# this is to suppress when we are the internal :)
target_compile_definitions(${STP_MAIN_ENGINE}
PRIVATE STP_IMPLEMENTATION
)