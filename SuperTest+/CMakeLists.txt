set(STP_UNIT_TEST "SuperTest+")
set(STP_CONSOLE_REPORTER "stp_console")

find_package(Catch2 3.3 REQUIRED CONFIG)

include(DownloadResource)
# benchmarking library by `nanobench`
downloadResource(
URL "https://raw.githubusercontent.com/martinus/nanobench/v4.3.11/src/include/nanobench.h"
FILENAME "nanobench-4.3.11/nanobench"
EXTENSION ".h"
PATH STP_NANOBENCH_DIR
)

# unit test codes
add_executable(${STP_UNIT_TEST}
# Test engine starter
STPTestMain.cpp
# Main test code
STPTest2DBiome.cpp
STPTest2DChunk.cpp
STPTestAlgebra.cpp
STPTestNearestNeighbour.cpp
STPTestRTC.cpp
STPTestTexture.cpp
STPTestUtility.cpp
# SuperAlgorithm+ test code
STPTestCmdParser.cpp
STPTestHistogram.cpp
STPTestINI.cpp
STPTestLexer.cpp
STPTestPermutation.cpp
STPTestRegLang.cpp
STPTestTDL.cpp
# Setup/
Setup/STPBenchmark.cpp
Setup/STPConsoleReporter.cpp
)

target_sources(${STP_UNIT_TEST} PRIVATE
STPTestInformation.h
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
PRIVATE Catch2::Catch2
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