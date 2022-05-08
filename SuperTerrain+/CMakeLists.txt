set(STP_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Template)
set(STP_TEMPLATE_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>TemplateInterface)
set(STP_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/CoreInterface)
set(STP_MODULE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ModuleInterface)
# config file information
set(SuperTerrain+CoreInclude ${STP_CORE_INCLUDE})
set(SuperAlgorithm+DeviceInclude ${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE}/Device)
set(SuperRealism+Include ${STP_MODULE_INCLUDE}/${STP_REALISM_ENGINE})
# a helper function to quickly setup config file
function(setupConfigTemplate BuildTarget TemplateInput TemplateOutput ConfigFilename)
	configure_file(${STP_TEMPLATE_DIR}/${TemplateInput} ${TemplateOutput} @ONLY)
	file(GENERATE
	OUTPUT ${STP_TEMPLATE_OUTPUT}/${BuildTarget}/${ConfigFilename}
	INPUT ${CMAKE_CURRENT_BINARY_DIR}/${TemplateOutput}
	)
endfunction()

# add SuperTerrain+, add headers first because it passes header variables out
add_subdirectory(${STP_CORE_INCLUDE}/${STP_MAIN_ENGINE})
add_subdirectory(Core)
# Add SuperAlgorithm+, add target first because headers are added via target_source
add_subdirectory(${STP_ALGORITHM_ENGINE})
add_subdirectory(${STP_MODULE_INCLUDE}/${STP_ALGORITHM_ENGINE})
# Add SuperRealism+
add_subdirectory(${STP_MODULE_INCLUDE}/${STP_REALISM_ENGINE})
add_subdirectory(${STP_REALISM_ENGINE})

# setup the template target
add_library(${STP_TEMPLATE_INCLUDE} INTERFACE)
# config file include
target_include_directories(${STP_TEMPLATE_INCLUDE}
INTERFACE ${STP_TEMPLATE_OUTPUT}
)

# dependencies
target_include_directories(${STP_MAIN_ENGINE}
PUBLIC ${STP_CORE_INCLUDE}
)
target_include_directories(${STP_ALGORITHM_HOST_ENGINE}
PUBLIC ${STP_MODULE_INCLUDE}
)
target_include_directories(${STP_ALGORITHM_DEVICE_ENGINE} # algorithm device engine do not expose any external header in its public headers
# we only include some define symbols from the core engine, no implementations are used
PRIVATE ${STP_CORE_INCLUDE} ${STP_MODULE_INCLUDE}
)
target_include_directories(${STP_REALISM_ENGINE}
PUBLIC ${STP_MODULE_INCLUDE}
)

# library dependencies
target_link_libraries(${STP_MAIN_ENGINE}
# we don't need to link SuperAlgorithm+, this is a library left for developers
# we do not plan to expose sqlite database in the header and to user, keep it private
PRIVATE ${STP_GLAD} SQLite::SQLite3
PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc glm::glm
)
target_link_libraries(${STP_ALGORITHM_HOST_ENGINE}
PUBLIC ${STP_MAIN_ENGINE} # main engine has all CUDA drivers linked as public
)
target_link_libraries(${STP_REALISM_ENGINE}
# template file contains private path to some shader source code
PRIVATE ${STP_GLAD} ${STP_TEMPLATE_INCLUDE}
PUBLIC ${STP_MAIN_ENGINE}
)