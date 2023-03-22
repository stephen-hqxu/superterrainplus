add_library(${STP_ALGORITHM_DEVICE_ENGINE} STATIC)
add_subdirectory(Private)
add_subdirectory(Public/${STP_ALGORITHM_DEVICE_ENGINE})

setupSourceGroup(${STP_ALGORITHM_DEVICE_ENGINE})

# config file for algorithm engine
set(SuperAlgorithm+DeviceInclude "${CMAKE_CURRENT_SOURCE_DIR}/Public")
setupConfigTemplate(${STP_ALGORITHM_DEVICE_ENGINE} "STPAlgorithmDeviceInfo")

target_include_directories(${STP_ALGORITHM_DEVICE_ENGINE}
PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public"
)