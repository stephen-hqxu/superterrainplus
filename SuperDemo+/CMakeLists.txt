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
)
if(${STP_ENGINE_BUILD_SHARED})
	# when building a shared library, GLAD.c is not exported by the engine so we need to compile it twice
	# when building a static library, GLAD.c object is included in the library so don't need to add it
	target_link_libraries(${STP_DEMO_PROGRAM}
	PRIVATE ${STP_GLAD}
	)
endif()

if(MSVC)
	# set VS debuging directory
	set_target_properties(${STP_DEMO_PROGRAM} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>)
endif()

target_link_directories(${STP_DEMO_PROGRAM}
PRIVATE ${STP_EXTERNAL_DIR}/lib
)
# all includes are included automatically since target include is public
# cuda related static libraries are also added by public target

target_link_libraries(${STP_DEMO_PROGRAM}
PRIVATE opengl32.lib glfw3.lib SglToolkit.lib SIMPLE.lib
PRIVATE ${STP_MAIN_ENGINE} ${STP_ERROR_ENGINE} ${STP_ALGORITHM_HOST_ENGINE}
)

# copy GLSL code to where the bin locates
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/GLSL
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>/GLSL
)

# copy resource
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/Resource
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>/Resource
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