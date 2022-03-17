# pre-compile GLAD so the object can be shared
add_library(${STP_GLAD} SHARED
src/glad.c
)

# GLAD is only a symbol importer, it's usually used along with GL (the definition of all drawing API functions)
target_link_libraries(${STP_GLAD}
INTERFACE OpenGL::GL
)
target_include_directories(${STP_GLAD}
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# instruct GLAD to use shared library mode for all linked library
target_compile_definitions(${STP_GLAD}
PUBLIC GLAD_GLAPI_EXPORT
)

# export symbols during build
set_target_properties(${STP_GLAD} PROPERTIES DEFINE_SYMBOL GLAD_GLAPI_EXPORT_BUILD)