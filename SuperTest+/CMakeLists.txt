set(STP_UNIT_TEST "SuperTest+")
set(STP_CONSOLE_REPORTER "stp_console")

# minimum catch2 v3.0
find_package(Catch2 3 REQUIRED CONFIG)

# unit test codes
add_executable(${STP_UNIT_TEST}
# Reporter/
Reporter/STPConsoleReporter.cpp
# Main test code
STPTest2DBiome.cpp
STPTest2DChunk.cpp
STPTestFreeSlip.cpp
STPTestMain.cpp
STPTestRTC.cpp
STPTestUtility.cpp
# SuperAlgorithm+ test code
STPTestHistogram.cpp
STPTestPermutation.cpp
# Prototype testing
STPPrototypeCachedConvolution.cpp
Compute/SobelFilter.cu
)
target_sources(${STP_UNIT_TEST} PRIVATE
Compute/SobelFilter.cuh
)

if(MSVC)
	set_target_properties(${STP_UNIT_TEST} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_UNIT_TEST}>)
endif()

# link with the main program that needs to be tested and catch2
target_link_libraries(${STP_UNIT_TEST}
PRIVATE ${STP_MAIN_ENGINE} ${STP_ALGORITHM_HOST_ENGINE}
PRIVATE Catch2::Catch2
)

# compile-time settings for catch2
target_compile_definitions(${STP_UNIT_TEST}
PRIVATE CATCH_CONFIG_CONSOLE_WIDTH=120
)

# catch2 with ctest integration
include(Catch)
catch_discover_tests(${STP_UNIT_TEST}
REPORTER ${STP_CONSOLE_REPORTER}
WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_UNIT_TEST}>
)

# copy data for testings
add_custom_command(
TARGET ${STP_UNIT_TEST} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/TestData
$<TARGET_FILE_DIR:${STP_UNIT_TEST}>/TestData
)