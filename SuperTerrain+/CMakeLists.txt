set(STP_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Template)
set(STP_TEMPLATE_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>TemplateInterface)

function(setupSourceGroup BuildTarget)
	# get all source files for this target
	get_target_property(TargetSource ${BuildTarget} SOURCES)

	source_group(
	TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${TargetSource}
	)
endfunction()

# a helper function to quickly setup config file for a given target based on an input template
function(setupConfigTemplate BuildTarget TemplateFilename)
	# derive config filename based on the given template filename
	set(TemplateInput "${TemplateFilename}.h.in")
	set(ConfigFilename "${TemplateFilename}.h")
	set(TemplateOutput "Template/${ConfigFilename}")

	configure_file(${STP_TEMPLATE_DIR}/${TemplateInput} ${TemplateOutput} @ONLY)
	file(GENERATE
	OUTPUT ${STP_TEMPLATE_OUTPUT}/${BuildTarget}/${ConfigFilename}
	INPUT ${CMAKE_CURRENT_BINARY_DIR}/${TemplateOutput}
	)
endfunction()

# SuperTerrain+
add_subdirectory(${STP_MAIN_ENGINE})
# SuperAlgorithm+
add_subdirectory(${STP_ALGORITHM_ENGINE}/Device)
add_subdirectory(${STP_ALGORITHM_ENGINE}/Host)
# SuperRealism+
add_subdirectory(${STP_REALISM_ENGINE})

# Template
add_library(${STP_TEMPLATE_INCLUDE} INTERFACE)
target_include_directories(${STP_TEMPLATE_INCLUDE}
INTERFACE ${STP_TEMPLATE_OUTPUT}
)

# algorithm device engine do not expose any external header in its public headers,
# we only include some define symbols from the core engine, no implementations are used
target_include_directories(${STP_ALGORITHM_DEVICE_ENGINE} 
PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/${STP_MAIN_ENGINE}/Public" "${CMAKE_CURRENT_SOURCE_DIR}/${STP_ALGORITHM_ENGINE}/Host/Public"
)

# resolve library dependencies
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
PUBLIC OptiX::OptiX
PUBLIC ${STP_MAIN_ENGINE}
)