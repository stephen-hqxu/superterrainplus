set(STP_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Template)
set(STP_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/CoreInterface)
set(STP_MODULE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ModuleInterface)
# add SuperTerrain+ source codes to target
add_subdirectory(Core)
add_subdirectory(${STP_CORE_INCLUDE}/${STP_MAIN_ENGINE})
# Add SuperAlgorithm+
add_subdirectory(${STP_ALGORITHM_ENGINE})
add_subdirectory(${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE})

# config file for algorithm engine
# tell the static library location so CUDA runtime compiler can find it easily (if developer wants to)
set(SuperTerrain+CoreInclude ${STP_CORE_INCLUDE})
set(SuperAlgorithm+DeviceInclude ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE}/Device)
# algorithm device info contains path to the static library, we need to output that based on configuration
configure_file(${STP_TEMPLATE_DIR}/STPAlgorithmDeviceInfo.h.in ${STP_ALGORITHM_DEFINE_INCLUDE}/STPAlgorithmDeviceInfoRaw.h @ONLY)
file(GENERATE
OUTPUT ${STP_ALGORITHM_DEFINE_INCLUDE}/STPAlgorithmDeviceInfo$<CONFIG>.h
INPUT ${STP_ALGORITHM_DEFINE_INCLUDE}/STPAlgorithmDeviceInfoRaw.h
)

# dependencies
target_include_directories(${STP_MAIN_ENGINE}
PRIVATE ${STP_MODULE_INCLUDE}
PUBLIC ${STP_CORE_INCLUDE}
)
target_include_directories(${STP_ALGORITHM_HOST_ENGINE}
PUBLIC ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE}
# info about the algorithm device engine, which may not be linked by the actual application if one chooses to use runtime comilation.
# however if algorithm device engine is linked (developer decides to not use RTC), then device info header is useless since the target has all info needed.
INTERFACE ${STP_ALGORITHM_DEFINE_INCLUDE}
)
target_include_directories(${STP_ALGORITHM_DEVICE_ENGINE} # algorithm device engine do not expose any external header in its public headers
# we only include some define symbols from the core engine, no implementations are used
PRIVATE ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE}
)

# library dependencies
target_link_libraries(${STP_MAIN_ENGINE}
# we don't need to link SuperAlgorithm+, this is a library left for developers
PRIVATE ${STP_GLAD} OpenGL::GL
PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc glm::glm
)
target_link_libraries(${STP_ALGORITHM_HOST_ENGINE}
PRIVATE ${STP_MAIN_ENGINE} # main engine has all CUDA drivers linked as public
)