set(STP_DEMO_PROGRAM "SuperDemo+")
# add all demo program code
# header and source code will be placed together
add_executable(${STP_DEMO_PROGRAM}
World/Layers/STPAllLayers.cpp
World/Biomes/STPBiome.cpp
World/Biomes/STPBiomefieldGenerator.cpp
World/STPProcedural2DINF.cpp
Helpers/STPStart.cpp
Helpers/STPTerrainParaLoader.cpp
Helpers/STPTextureStorage.cpp
World/STPWorldManager.cpp

World/Biomes/STPMultiHeightGenerator.cu
)

target_link_directories(${STP_DEMO_PROGRAM}
PRIVATE ${STP_EXTERNAL_DIR}/lib
)
target_include_directories(${STP_DEMO_PROGRAM}
PRIVATE ${CMAKE_SOURCE_DIR}/SuperTerrain+/CoreInterface ${CMAKE_SOURCE_DIR}/SuperTerrain+/ModuleInterface
)

target_link_libraries(${STP_DEMO_PROGRAM}
PRIVATE opengl32.lib glfw3.lib SglToolkit.lib SIMPLE.lib
PRIVATE SuperTerrain+ SuperError+ SuperAlgorithm+
)

# copy GLSL code to where the bin locates
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/GLSL
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>/GLSL
)

# copy dll
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${STP_EXTERNAL_DIR}/dll/glfw3.dll
${STP_EXTERNAL_DIR}/dll/SIMPLE.dll
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>
)

# copy ini
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${CMAKE_CURRENT_SOURCE_DIR}/Engine.ini
${CMAKE_CURRENT_SOURCE_DIR}/Biome.ini
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>
)