set(MAIN_ENGINE_SOURCE
STPBiomeFactory.cpp
STPChunk.cpp
STPChunkManager.cpp
STPChunkProvider.cpp
STPChunkStorage.cpp
STPDiversityGeneratorRTC.cpp
STPEngineInitialiser.cpp
STPLayer.cpp
STPLayerCache.cpp
STPLayerManager.cpp
STPSeedMixer.cpp
STPThreadPool.cpp

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

# glad.c contains API entries to OpenGL functions
target_sources(
${STP_MAIN_ENGINE}
PRIVATE ${STP_EXTERNAL_DIR}/include/glad/glad.c
)

# link lib
target_link_libraries(${STP_MAIN_ENGINE}
PRIVATE opengl32.lib
)