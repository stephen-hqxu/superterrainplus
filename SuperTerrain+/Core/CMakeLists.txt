set(MAIN_ENGINE_SOURCE
STPBiomeFactory.cpp
STPChunk.cpp
STPChunkManager.cpp
STPChunkProvider.cpp
STPChunkStorage.cpp
STPDiversityGeneratorRTC.cpp
STPEngineInitialiser.cpp
STPEnvironment.cpp
STPFreeSlipTextureBuffer.cpp
STPLayer.cpp
STPLayerCache.cpp
STPLayerManager.cpp
STPMemoryPool.cpp
STPSeedMixer.cpp
STPSmartStream.cpp
STPThreadPool.cpp

STPFreeSlipGenerator.cu
STPFreeSlipManager.cu
STPHeightfieldGenerator.cu
STPRainDrop.cu
STPRainDropSetting.cu
)

if(${STP_ENGINE_BUILD_SHARED})
	add_library(${STP_MAIN_ENGINE} SHARED ${MAIN_ENGINE_SOURCE})
	# set export preprocessor
	set_target_properties(${STP_MAIN_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERTERRAINPLUS_EXPORTS)
else()
	add_library(${STP_MAIN_ENGINE} STATIC ${MAIN_ENGINE_SOURCE})
	set_target_properties(${STP_MAIN_ENGINE} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ${STP_DEVICELINK_STATIC})
endif()

# tell the compiler we are the source engine builder
# there are some preprocessors in the source code to warn or prevent external environment from including internal headers
# this is to suppress when we are the internal :)
target_compile_definitions(${STP_MAIN_ENGINE}
PRIVATE STP_IMPLEMENTATION
)