add_library(${STP_MAIN_ENGINE} SHARED)
add_subdirectory(Private)
add_subdirectory(Public/${STP_MAIN_ENGINE})

setupSourceGroup(${STP_MAIN_ENGINE})

set(SuperTerrain+Include "${CMAKE_CURRENT_SOURCE_DIR}/Public")
setupConfigTemplate(${STP_MAIN_ENGINE} "STPCoreInfo")
setupConfigTemplate(${STP_MAIN_ENGINE} "STPVersion")

target_include_directories(${STP_MAIN_ENGINE}
PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public"
)
# set export preprocessor
set_target_properties(${STP_MAIN_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERTERRAINPLUS_EXPORTS)
# tell the compiler we are the source engine builder
# there are some preprocessors in the source code to warn or prevent external environment from including internal headers
# this is to suppress when we are the internal :)
target_compile_definitions(${STP_MAIN_ENGINE}
PRIVATE STP_IMPLEMENTATION
)