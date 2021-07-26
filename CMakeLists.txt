cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SuperTerrain+
VERSION 0.4.5
LANGUAGES C CXX CUDA
DESCRIPTION "A real-time procedural 3D infinite terrain engine with geographical features and photorealistic rendering."
HOMEPAGE_URL "https://github.com/stephen-hqxu/superterrainplus"
)
# addition CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

# get list of options
include(OptionList)
# setup compilers
include(SetupEnvironment)

# add main STP engine
add_subdirectory(SuperTerrain+)

# copy dll to output folder
file(GLOB STP_DLL_DEPENDENCIES ${CMAKE_SOURCE_DIR}/external/dll)
file(COPY ${STP_DLL_DEPENDENCIES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})