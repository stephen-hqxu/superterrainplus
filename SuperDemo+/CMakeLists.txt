set(STP_DEMO_PROGRAM "SuperDemo+")

# download external libraries and assets
include(DownloadResource)
# image reader by `stb`
downloadResource(
URL "https://raw.githubusercontent.com/nothings/stb/3ecc60f25ae1391cf6434578ece782afa1458b56/stb_image.h"
FILENAME "stb_image-2.28/stb_image"
EXTENSION ".h"
PATH STP_STB_IMAGE_DIR
)
# resource
downloadResource(
URL "https://github.com/stephen-hqxu/superterrainplus/releases/download/v0.13.7/Resource.tar.gz"
FILENAME "Resource-0.13.7"
EXTENSION ".tar.gz"
LOCATION STP_RESOURCE_DIR
EXTRACT
)

# minimum GLFW requirement v3.0
find_package(glfw3 3 REQUIRED CONFIG)

# add all demo program code
# header and source code will be placed together
add_executable(${STP_DEMO_PROGRAM}
STPMasterRenderer.cpp
STPStart.cpp
# /Helpers
Helpers/STPCommandLineOption.cpp
Helpers/STPTerrainParaLoader.cpp
Helpers/STPTextureStorage.cpp
# /World/Biomes
World/Biomes/STPBiomefieldGenerator.cpp
World/Biomes/STPCommonCompiler.cpp
World/Biomes/STPSplatmapGenerator.cpp
# /World/Layers
World/Layers/STPAllLayers.cpp
# /World
World/STPWorldManager.cpp
)

target_sources(${STP_DEMO_PROGRAM} PRIVATE
STPMasterRenderer.h
# /Helpers
Helpers/STPCommandLineOption.h
Helpers/STPTerrainParaLoader.h
Helpers/STPTextureStorage.h
# /World/Biomes
World/Biomes/STPBiome.hpp
World/Biomes/STPBiomefieldGenerator.h
World/Biomes/STPBiomeProperty.hpp
World/Biomes/STPBiomeRegistry.h
World/Biomes/STPCommonCompiler.h
World/Biomes/STPSplatmapGenerator.h
# /World/Layers
World/Layers/STPAllLayers.h
World/Layers/STPBaseBiomeLayer.h
World/Layers/STPClimateLayer.h
World/Layers/STPContinentLayer.h
World/Layers/STPCrossLayer.h
World/Layers/STPDeepOceanLayer.h
World/Layers/STPEaseEdgeLayer.h
World/Layers/STPEdgeBiomeLayer.h
World/Layers/STPHillsLayer.h
World/Layers/STPIslandLayer.h
World/Layers/STPLandLayer.h
World/Layers/STPNoiseLayer.h
World/Layers/STPNoiseToRiverLayer.h
World/Layers/STPOceanTemperatureLayer.h
World/Layers/STPRiverErodeLayer.h
World/Layers/STPRiverMixLayer.h
World/Layers/STPScaleLayer.h
World/Layers/STPSmoothScaleLayer.h
World/Layers/STPVoronoiLayer.h
World/Layers/STPXCrossLayer.h
# /World
World/STPWorldManager.h
World/STPWorldManager.inl
)

if(MSVC)
	# set VS debugging directory
	set_target_properties(${STP_DEMO_PROGRAM} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_DEMO_PROGRAM}>)
endif()

target_include_directories(${STP_DEMO_PROGRAM}
PRIVATE ${STP_STB_IMAGE_DIR}
)

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