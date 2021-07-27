# TODO: add shared library option
add_library(${STP_MAIN_ENGINE} STATIC
# main engine src
STPBiomeFactory.cpp
STPChunk.cpp
STPChunkManager.cpp
STPChunkProvider.cpp
STPChunkStorage.cpp
STPDiversityGeneratorRTC.cpp
STPLayer.cpp
STPLayerCache.cpp
STPLayerManager.cpp
STPSeedMixer.cpp
STPThreadPool.cpp

STPHeightfieldGenerator.cu
STPRainDrop.cu
STPRainDropSettings.cu
)

# glad.c contains API entries to OpenGL functions
target_sources(
${STP_MAIN_ENGINE}
PRIVATE ${STP_EXTERNAL_DIR}/include/glad/glad.c
)

# link lib
target_link_libraries(${STP_MAIN_ENGINE}
PRIVATE opengl32.lib
)