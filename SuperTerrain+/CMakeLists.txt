set(STP_MAIN_ENGINE "SuperTerrain+")
set(STP_ERROR_ENGINE "SuperError+")
set(STP_ALGORITHM_ENGINE "SuperAlgorithm+")
# add SuperTerrain+ source codes to target
add_subdirectory(src)
# add SuperError+ source code
add_subdirectory(${STP_ERROR_ENGINE})
# Add SuperAlgorithm+
add_subdirectory(${STP_ALGORITHM_ENGINE})

# dependencies
target_include_directories(${STP_MAIN_ENGINE}
PRIVATE ${STP_EXTERNAL_DIR}/include
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/${STP_ERROR_ENGINE} ${CMAKE_CURRENT_SOURCE_DIR}/${STP_ALGORITHM_ENGINE}
)
target_include_directories(${STP_ALGORITHM_ENGINE}
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${STP_ERROR_ENGINE} ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# link error handling engine target
target_link_libraries(${STP_MAIN_ENGINE}
# TODO: After we split demo and the actual program, we don't need to link SuperAlgorithm+, this is a library left for developer
PRIVATE ${STP_ERROR_ENGINE} ${STP_ALGORITHM_ENGINE}
)