set(STP_MAIN_ENGINE "SuperTerrain+")
set(STP_ERROR_ENGINE "SuperError+")
set(STP_ALGORITHM_ENGINE "SuperAlgorithm+")
# add SuperTerrain+ source codes to target
add_subdirectory(Core)
# add SuperError+ source code
add_subdirectory(${STP_ERROR_ENGINE})
# Add SuperAlgorithm+
add_subdirectory(${STP_ALGORITHM_ENGINE})

set(STP_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Template)
set(STP_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/CoreInterface)
set(STP_MODULE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ModuleInterface)

# tell the static library location so CUDA runtime compiler can find it easily (if developer wants to)
get_target_property(SuperAlgorithm+Lib ${STP_ALGORITHM_ENGINE} ARCHIVE_OUTPUT_DIRECTORY)
set(SuperAlgorithm+Include ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE})
# config file for SuperTerrain+ engine
configure_file(${STP_TEMPLATE_DIR}/STPDefine.h ${STP_CORE_INCLUDE}/STPDefine.h @ONLY)

# dependencies
target_include_directories(${STP_MAIN_ENGINE}
PUBLIC ${STP_EXTERNAL_DIR}/include ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE}
)
target_include_directories(${STP_ERROR_ENGINE}
PRIVATE ${STP_MODULE_INCLUDE}/${STP_ERROR_ENGINE}
)
target_include_directories(${STP_ALGORITHM_ENGINE}
PRIVATE ${STP_MODULE_INCLUDE}/${STP_ERROR_ENGINE} ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE} ${STP_CORE_INCLUDE}
)

# link error handling engine target
target_link_libraries(${STP_MAIN_ENGINE}
# we don't need to link SuperAlgorithm+, this is a library left for developers
PRIVATE ${STP_ERROR_ENGINE}
)