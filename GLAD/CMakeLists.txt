# pre-compile GLAD so the object can be shared
add_library(${STP_GLAD} OBJECT
glad/glad.c
)

target_include_directories(${STP_GLAD}
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)