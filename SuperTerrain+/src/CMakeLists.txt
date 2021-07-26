add_executable(${STP_MAIN_ENGINE}
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
# demo program src, I will separate demo program into another target or project later
STPAllLayers.cpp
STPBiome.cpp
STPBiomefieldGenerator.cpp
STPProcedural2DINF.cpp
STPStart.cpp
STPTerrainParaLoader.cpp
STPTextureStorage.cpp
STPWorldManager.cpp

STPMultiHeightGenerator.cu
)

target_sources(
${STP_MAIN_ENGINE}
PRIVATE ${STP_EXTERNAL_DIR}/include/glad/glad.c
)

# target dependencies
target_link_directories(${STP_MAIN_ENGINE}
PRIVATE ${STP_EXTERNAL_DIR}/lib
)
# link lib
target_link_libraries(${STP_MAIN_ENGINE}
PRIVATE opengl32.lib glfw3.lib SglToolkit.lib SIMPLE.lib
)   

# copy GLSL code to where the bin locates
add_custom_command(
TARGET ${STP_MAIN_ENGINE} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/GLSL
$<TARGET_FILE_DIR:${STP_MAIN_ENGINE}>/GLSL
)

# copy dll
add_custom_command(
TARGET ${STP_MAIN_ENGINE} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${STP_EXTERNAL_DIR}/dll/glfw3.dll
${STP_EXTERNAL_DIR}/dll/SIMPLE.dll
$<TARGET_FILE_DIR:${STP_MAIN_ENGINE}>
)

# copy ini
add_custom_command(
TARGET ${STP_MAIN_ENGINE} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${CMAKE_SOURCE_DIR}/Engine.ini
${CMAKE_SOURCE_DIR}/Biome.ini
$<TARGET_FILE_DIR:${STP_MAIN_ENGINE}>
)