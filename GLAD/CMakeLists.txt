project(GLAD
VERSION 0.1.35
LANGUAGES C
DESCRIPTION "Multi-Language Vulkan/GL/GLES/EGL/GLX/WGL Loader-Generator based on the official specs."
HOMEPAGE_URL "https://github.com/Dav1dde/glad"
)

# setup GLAD library source files
include(DownloadResource)
downloadResource(
URL "https://github.com/stephen-hqxu/superterrainplus/releases/download/v0.13.7/GLAD.tar.gz"
FILENAME "GLAD-${PROJECT_VERSION}"
EXTENSION ".tar.gz"
LOCATION STP_GLAD_DIR
EXTRACT
)

# pre-compile GLAD so the object can be shared
add_library(${STP_GLAD} SHARED
${STP_GLAD_DIR}/src/glad.c
)

target_link_libraries(${STP_GLAD}
INTERFACE OpenGL::GL
)
target_include_directories(${STP_GLAD}
PUBLIC ${STP_GLAD_DIR}/include
)

# instruct GLAD to use shared library mode for all linked library
# have to use shared library so different build targets can share the context
target_compile_definitions(${STP_GLAD}
PUBLIC GLAD_GLAPI_EXPORT
)
set_target_properties(${STP_GLAD} PROPERTIES DEFINE_SYMBOL GLAD_GLAPI_EXPORT_BUILD)