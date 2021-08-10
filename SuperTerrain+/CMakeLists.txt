set(STP_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Template)
set(STP_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/CoreInterface)
set(STP_MODULE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ModuleInterface)
# add SuperTerrain+ source codes to target
add_subdirectory(Core)
add_subdirectory(${STP_CORE_INCLUDE})
# add SuperError+ source code
add_subdirectory(${STP_ERROR_ENGINE})
add_subdirectory(${STP_MODULE_INCLUDE}/${STP_ERROR_ENGINE})
# Add SuperAlgorithm+
add_subdirectory(${STP_ALGORITHM_ENGINE})
add_subdirectory(${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE})

# config file for main engine
if(${STP_ENGINE_BUILD_SHARED})
	# expose export/import symbols on shared library build
	set(SUPERTERRAINPLUS_USE_SHARED ON)
endif()
# core define include dir is set in core sub-dir
configure_file(${STP_TEMPLATE_DIR}/STPCoreDefine.h.in ${STP_CORE_DEFINE_INCLUDE}/STPCoreDefine.h @ONLY)
# config file for algorithm engine
# tell the static library location so CUDA runtime compiler can find it easily (if developer wants to)
set(SuperAlgorithm+DeviceInclude ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE}/Device)
# algorithm define include dir is set in algorithm sub-dir
configure_file(${STP_TEMPLATE_DIR}/STPAlgorithmDefine.h.in ${STP_ALGORITHM_DEFINE_INCLUDE}/STPAlgorithmDefine.h COPYONLY)
# algorithm device info contains path to the static library, we need to output that based on configuration
configure_file(${STP_TEMPLATE_DIR}/STPAlgorithmDeviceInfo.h.in ${STP_ALGORITHM_DEFINE_INCLUDE}/STPAlgorithmDeviceInfoRaw.h @ONLY)
file(GENERATE
OUTPUT ${STP_ALGORITHM_DEFINE_INCLUDE}/STPAlgorithmDeviceInfo$<CONFIG>.h
INPUT ${STP_ALGORITHM_DEFINE_INCLUDE}/STPAlgorithmDeviceInfoRaw.h
)

# dependencies
target_include_directories(${STP_MAIN_ENGINE}
PUBLIC ${STP_EXTERNAL_DIR}/include ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE} ${STP_CORE_DEFINE_INCLUDE}
)
target_include_directories(${STP_ERROR_ENGINE}
PRIVATE ${STP_MODULE_INCLUDE}/${STP_ERROR_ENGINE}
)
target_include_directories(${STP_ALGORITHM_HOST_ENGINE}
PRIVATE ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE}
PUBLIC ${STP_ALGORITHM_DEFINE_INCLUDE}
)
target_include_directories(${STP_ALGORITHM_DEVICE_ENGINE}
PRIVATE ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE}/Device
)

# library dependencies
target_link_libraries(${STP_MAIN_ENGINE}
# we don't need to link SuperAlgorithm+, this is a library left for developers
PRIVATE ${STP_ERROR_ENGINE} ${STP_GLAD} opengl32.lib
PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
)
target_link_libraries(${STP_ALGORITHM_HOST_ENGINE}
PRIVATE ${STP_ERROR_ENGINE} CUDA::cudart
)
# device engine doesn't require lib
target_link_libraries(${STP_ERROR_ENGINE}
PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
)