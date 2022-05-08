add_library(${STP_MAIN_ENGINE} SHARED
STPAlgebra.cpp
STPBiomeFactory.cpp
STPChunk.cpp
STPDeviceErrorHandler.cpp
STPEngineInitialiser.cpp
STPEnvironment.cpp
STPErosionBrushGenerator.cpp
STPException.cpp
STPFile.cpp
STPFreeSlipTextureBuffer.cpp
STPHeightfieldGenerator.cpp
STPLayer.cpp
STPLayerCache.cpp
STPLayerManager.cpp
STPMemoryPool.cpp
STPNullablePrimitive.cpp
STPRuntimeCompilable.cpp
STPSmartStream.cpp
STPTextureDatabase.cpp
STPTextureDefinitionLanguage.cpp
STPTextureFactory.cpp
STPThreadPool.cpp
STPWorldPipeline.cpp

STPHeightfieldKernel.cu
STPRainDrop.cu
STPRainDropSetting.cu
)
# set export preprocessor
set_target_properties(${STP_MAIN_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERTERRAINPLUS_EXPORTS)

# tell the compiler we are the source engine builder
# there are some preprocessors in the source code to warn or prevent external environment from including internal headers
# this is to suppress when we are the internal :)
target_compile_definitions(${STP_MAIN_ENGINE}
PRIVATE STP_IMPLEMENTATION
)