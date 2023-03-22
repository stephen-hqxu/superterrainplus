add_library(${STP_ALGORITHM_HOST_ENGINE} SHARED)
add_subdirectory(Private)
add_subdirectory(Public/${STP_ALGORITHM_HOST_ENGINE})

setupSourceGroup(${STP_ALGORITHM_HOST_ENGINE})

set(SuperAlgorithm+HostInclude "${CMAKE_CURRENT_SOURCE_DIR}/Public")
setupConfigTemplate(${STP_ALGORITHM_HOST_ENGINE} "STPAlgorithmHostInfo")

target_include_directories(${STP_ALGORITHM_HOST_ENGINE}
PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public"
)
set_target_properties(${STP_ALGORITHM_HOST_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERALGORITHMPLUS_HOST_EXPORTS)