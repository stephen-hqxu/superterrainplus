set(STP_UNIT_TEST "SuperTest+")

# minimum catch2 v3.0
find_package(Catch2 3 REQUIRED CONFIG)

# unit test codes
add_executable(${STP_UNIT_TEST}
# Reporter/
Reporter/STPConsoleReporter.cpp
# Main test code
STPTest2DChunk.cpp
STPTestUtility.cpp
)

# link with the main program that needs to be tested and catch2
target_link_libraries(${STP_UNIT_TEST}
PRIVATE ${STP_MAIN_ENGINE}
PRIVATE Catch2::Catch2WithMain
)

# compile-time settings for catch2
target_compile_definitions(${STP_UNIT_TEST}
PRIVATE CATCH_CONFIG_CONSOLE_WIDTH=100
)

# catch2 with ctest integration
include(Catch)
catch_discover_tests(${STP_UNIT_TEST}
REPORTER stp_console
)