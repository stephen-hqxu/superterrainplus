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

if(${STP_ENGINE_BUILD_SHARED})
	# expose export/import symbols on shared library build
	set(SuperTerrain+Shared "")
else()
	# hide export/import symbols on static library build
	set(SuperTerrain+Shared "//")
endif()
# tell the static library location so CUDA runtime compiler can find it easily (if developer wants to)
get_target_property(SuperAlgorithm+Lib ${STP_ALGORITHM_DEVICE_ENGINE} ARCHIVE_OUTPUT_DIRECTORY)
set(SuperAlgorithm+Include ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE})
# config files for engine
configure_file(${STP_TEMPLATE_DIR}/STPCoreDefine.h.in ${STP_CORE_INCLUDE}/STPCoreDefine.h @ONLY)
configure_file(${STP_TEMPLATE_DIR}/STPAlgorithmDefine.h.in ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE}/STPAlgorithmDefine.h @ONLY)

# dependencies
target_include_directories(${STP_MAIN_ENGINE}
PUBLIC ${STP_EXTERNAL_DIR}/include ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE}
)
target_include_directories(${STP_ERROR_ENGINE}
PRIVATE ${STP_MODULE_INCLUDE}/${STP_ERROR_ENGINE}
)
target_include_directories(${STP_ALGORITHM_HOST_ENGINE}
PRIVATE ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE}
)
target_include_directories(${STP_ALGORITHM_DEVICE_ENGINE}
PRIVATE ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE}/Device
)

# library dependencies
target_link_libraries(${STP_MAIN_ENGINE}
# we don't need to link SuperAlgorithm+, this is a library left for developers
PRIVATE ${STP_ERROR_ENGINE} ${STP_GLAD} opengl32.lib
)
target_link_libraries(${STP_ALGORITHM_HOST_ENGINE}
PRIVATE ${STP_ERROR_ENGINE}
PUBLIC CUDA::cudart
)
# device engine doesn't require lib
target_link_libraries(${STP_ERROR_ENGINE}
PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
)