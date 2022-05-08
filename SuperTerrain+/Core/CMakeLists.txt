set(STP_MAIN_TERRAIN
# /Terrain
Terrain/STPBiomeFactory.cpp
Terrain/STPChunk.cpp
Terrain/STPFreeSlipTextureBuffer.cpp
Terrain/STPHeightfieldGenerator.cpp
Terrain/STPLayer.cpp
Terrain/STPLayerCache.cpp
Terrain/STPLayerManager.cpp
Terrain/STPTextureDatabase.cpp
Terrain/STPTextureDefinitionLanguage.cpp
Terrain/STPTextureFactory.cpp
Terrain/STPWorldPipeline.cpp
)
set(STP_MAIN_SOURCE
STPAlgebra.cpp
STPDeviceErrorHandler.cpp
STPEngineInitialiser.cpp
STPEnvironment.cpp
STPErosionBrushGenerator.cpp
STPException.cpp
STPFile.cpp
STPMemoryPool.cpp
STPNullablePrimitive.cpp
STPRuntimeCompilable.cpp
STPSmartStream.cpp
STPThreadPool.cpp

STPHeightfieldKernel.cu
STPRainDrop.cu
STPRainDropSetting.cu
)

add_library(${STP_MAIN_ENGINE} SHARED
${STP_MAIN_TERRAIN}
${STP_MAIN_SOURCE}

${STP_MAIN_ENGINE_INTERFACE}
)
# add source file filter
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}
PREFIX Source
FILES ${STP_MAIN_TERRAIN} ${STP_MAIN_SOURCE}
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