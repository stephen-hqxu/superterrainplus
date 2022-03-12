set(STP_EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/External)
set(STP_DEMO_PROGRAM "SuperDemo+")

# minimum GLFW requirement v3.0
find_package(glfw3 3 REQUIRED CONFIG)
# and include our home-made libraries ;)
find_package(SIMPLE 3 REQUIRED CONFIG)

# add all demo program code
# header and source code will be placed together
add_executable(${STP_DEMO_PROGRAM}
World/Layers/STPAllLayers.cpp
World/Biomes/STPBiome.cpp
World/Biomes/STPBiomefieldGenerator.cpp
World/Biomes/STPCommonCompiler.cpp
World/Biomes/STPSplatmapGenerator.cpp
Helpers/STPTerrainParaLoader.cpp
Helpers/STPTextureStorage.cpp
World/STPWorldManager.cpp
STPStart.cpp
)

if(MSVC)
	# set VS debuging directory
	set_target_properties(${STP_DEMO_PROGRAM} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>)
endif()

# cuda related static libraries are added by public target
target_include_directories(${STP_DEMO_PROGRAM}
PRIVATE ${STP_EXTERNAL_DIR}
)

# when building a shared library, GLAD.c is not exported by the engine so we need to compile it twice
target_link_libraries(${STP_DEMO_PROGRAM}
PRIVATE glfw SIMPLE::SIMPLE
PRIVATE ${STP_GLAD} ${STP_MAIN_ENGINE} ${STP_ALGORITHM_HOST_ENGINE} ${STP_REALISM_ENGINE} ${STP_TEMPLATE_INCLUDE}
)

# copy runtime generator script
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/Script
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>/Script
)

# copy resource
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/Resource
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>/Resource
)

# copy ini
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${CMAKE_CURRENT_SOURCE_DIR}/Engine.ini
${CMAKE_CURRENT_SOURCE_DIR}/Biome.ini
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>
)