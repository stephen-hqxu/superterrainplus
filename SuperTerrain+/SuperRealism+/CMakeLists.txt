add_library(${STP_REALISM_ENGINE} SHARED)
add_subdirectory(Private)
add_subdirectory(Public/${STP_REALISM_ENGINE})

setupSourceGroup(${STP_REALISM_ENGINE})

# config file for the realism rendering engine
set(SuperRealism+ShaderPath "${CMAKE_CURRENT_SOURCE_DIR}/Public/${STP_REALISM_ENGINE}/Shader")
set(SuperRealism+OptiXSDK "${OptiX_INSTALL_DIR}/SDK")
setupConfigTemplate(${STP_REALISM_ENGINE} "STPRealismInfo")

target_include_directories(${STP_REALISM_ENGINE}
PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public"
)
set_target_properties(${STP_REALISM_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERREALISMPLUS_EXPORTS)
target_compile_definitions(${STP_REALISM_ENGINE}
PRIVATE STP_REALISM_IMPLEMENTATION
)