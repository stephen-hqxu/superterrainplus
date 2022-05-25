set(STP_DEMO_PROGRAM "SuperDemo+")

# download external libraries and assets
include(DownloadResource)
# some external libraries
downloadResource(
URL "https://github.com/stephen-hqxu/superterrainplus/releases/download/v0.13.7/External.tar.gz"
FILENAME "External"
EXTENSION ".tar.gz"
LOCATION STP_EXTERNAL_DIR
EXTRACT
)
# resource
downloadResource(
URL "https://github.com/stephen-hqxu/superterrainplus/releases/download/v0.13.7/Resource.tar.gz"
FILENAME "Resource"
EXTENSION ".tar.gz"
LOCATION STP_RESOURCE_DIR
EXTRACT
)

# minimum GLFW requirement v3.0
find_package(glfw3 3 REQUIRED CONFIG)

# add all demo program code
# header and source code will be placed together
add_executable(${STP_DEMO_PROGRAM}
World/Layers/STPAllLayers.cpp
World/Biomes/STPBiomefieldGenerator.cpp
World/Biomes/STPCommonCompiler.cpp
World/Biomes/STPSplatmapGenerator.cpp
Helpers/STPTerrainParaLoader.cpp
Helpers/STPTextureStorage.cpp
World/STPWorldManager.cpp
STPStart.cpp
)

if(MSVC)
	# set VS debugging directory
	set_target_properties(${STP_DEMO_PROGRAM} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>)
endif()

# CUDA related static libraries are added by public target
target_include_directories(${STP_DEMO_PROGRAM}
PRIVATE ${STP_EXTERNAL_DIR}
)

# when building a shared library, GLAD.c is not exported by the engine so we need to compile it twice
target_link_libraries(${STP_DEMO_PROGRAM}
PRIVATE glfw
PRIVATE ${STP_GLAD} ${STP_MAIN_ENGINE} ${STP_ALGORITHM_HOST_ENGINE} ${STP_REALISM_ENGINE} ${STP_TEMPLATE_INCLUDE}
)

# copy runtime generator script
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/Script
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>/Script
)

add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${CMAKE_CURRENT_SOURCE_DIR}/World/Biomes/STPBiomeProperty.hpp
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>
)

# copy resource
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_directory
${STP_RESOURCE_DIR}
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>/Resource
)

# copy INI
add_custom_command(
TARGET ${STP_DEMO_PROGRAM} POST_BUILD      
COMMAND ${CMAKE_COMMAND} -E copy_if_different
${CMAKE_CURRENT_SOURCE_DIR}/Engine.ini
${CMAKE_CURRENT_SOURCE_DIR}/Biome.ini
$<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>
)