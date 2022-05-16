project(GLAD
LANGUAGES C
DESCRIPTION "Multi-Language Vulkan/GL/GLES/EGL/GLX/WGL Loader-Generator based on the official specs."
HOMEPAGE_URL "https://github.com/Dav1dde/glad"
)

# pre-compile GLAD so the object can be shared
add_library(${STP_GLAD} SHARED
src/glad.c
)

target_link_libraries(${STP_GLAD}
INTERFACE OpenGL::GL
)
target_include_directories(${STP_GLAD}
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# instruct GLAD to use shared library mode for all linked library
# have to use shared library so different build targets can share the context
target_compile_definitions(${STP_GLAD}
PUBLIC GLAD_GLAPI_EXPORT
)
set_target_properties(${STP_GLAD} PROPERTIES DEFINE_SYMBOL GLAD_GLAPI_EXPORT_BUILD)