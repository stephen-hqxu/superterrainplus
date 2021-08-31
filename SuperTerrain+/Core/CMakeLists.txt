add_library(${STP_MAIN_ENGINE} SHARED
STPBiomeFactory.cpp
STPChunk.cpp
STPChunkManager.cpp
STPChunkProvider.cpp
STPChunkStorage.cpp
STPDeviceErrorHandler.cpp
STPDiversityGeneratorRTC.cpp
STPEngineInitialiser.cpp
STPEnvironment.cpp
STPException.cpp
STPFreeSlipTextureBuffer.cpp
STPLayer.cpp
STPLayerCache.cpp
STPLayerManager.cpp
STPMemoryPool.cpp
STPSeedMixer.cpp
STPSmartStream.cpp
STPThreadPool.cpp

STPFreeSlipGenerator.cu
STPHeightfieldGenerator.cu
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