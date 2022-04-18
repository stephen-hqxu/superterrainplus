cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SuperTerrain+
VERSION 0.13.0
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

# set some useful variables of our main engine
set(STP_GLAD "GLAD")
# a target with config file include directory
set(STP_TEMPLATE_INCLUDE "TemplateInterface")
set(STP_MAIN_ENGINE "SuperTerrain+")
set(STP_ALGORITHM_ENGINE "SuperAlgorithm+")
set(STP_ALGORITHM_HOST_ENGINE "SuperAlgorithm+Host")
set(STP_ALGORITHM_DEVICE_ENGINE "SuperAlgorithm+Device")
set(STP_REALISM_ENGINE "SuperRealism+")

# pre-compile GLAD
add_subdirectory(GLAD)
# add main STP engine
add_subdirectory(SuperTerrain+)
# add demo program to show how STP engine works
if(${STP_BUILD_DEMO})
	add_subdirectory(SuperDemo+)
endif()
# add unit test program
if(${STP_BUILD_TEST})
	enable_testing()
	add_subdirectory(SuperTest+)
endif()