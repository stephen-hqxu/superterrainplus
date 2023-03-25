set(STP_UNIT_TEST "SuperTest+")
set(STP_CONSOLE_REPORTER "stp_console")

find_package(Catch2 3.3 REQUIRED CONFIG)
find_package(nanobench REQUIRED CONFIG)

# unit test codes
add_executable(${STP_UNIT_TEST})

set(STP_UNIT_TEST_HEADER
STPTestInformation.h
)

set(STP_UNIT_TEST_SOURCE
STPTestMain.cpp
# /Setup
Setup/STPConsoleReporter.cpp
# /SuperAlgorithm+
SuperAlgorithm+/STPTestCmdParser.cpp
SuperAlgorithm+/STPTestHistogram.cpp
SuperAlgorithm+/STPTestINI.cpp
SuperAlgorithm+/STPTestLexer.cpp
SuperAlgorithm+/STPTestPermutation.cpp
SuperAlgorithm+/STPTestRegLang.cpp
SuperAlgorithm+/STPTestTDL.cpp
# /SuperTerrain+
SuperTerrain+/STPTest2DBiome.cpp
SuperTerrain+/STPTest2DChunk.cpp
SuperTerrain+/STPTestAlgebra.cpp
SuperTerrain+/STPTestNearestNeighbour.cpp
SuperTerrain+/STPTestRTC.cpp
SuperTerrain+/STPTestTexture.cpp
SuperTerrain+/STPTestUtility.cpp
)

target_sources(${STP_UNIT_TEST} PRIVATE
${STP_UNIT_TEST_HEADER}
${STP_UNIT_TEST_SOURCE}
)

# add file filter
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}
PREFIX "Interface"
FILES ${STP_UNIT_TEST_HEADER}
)
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}
PREFIX "Implementation"
FILES ${STP_UNIT_TEST_SOURCE}
)

if(MSVC)
	set_target_properties(${STP_UNIT_TEST} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_UNIT_TEST}>)
endif()

target_include_directories(${STP_UNIT_TEST}
PRIVATE ${STP_NANOBENCH_DIR}
)

# link with the main program that needs to be tested and catch2
target_link_libraries(${STP_UNIT_TEST}
PRIVATE ${STP_MAIN_ENGINE} ${STP_ALGORITHM_HOST_ENGINE}
PRIVATE Catch2::Catch2 nanobench::nanobench
)

# catch2 with CTest integration
include(Catch)
catch_discover_tests(${STP_UNIT_TEST}
REPORTER ${STP_CONSOLE_REPORTER}
WORKING_DIRECTORY $<TARGET_FILE_DIR:${STP_UNIT_TEST}>
)

# copy data for test
add_custom_command(
TARGET ${STP_UNIT_TEST} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_CURRENT_SOURCE_DIR}/TestData
$<TARGET_FILE_DIR:${STP_UNIT_TEST}>/TestData
)