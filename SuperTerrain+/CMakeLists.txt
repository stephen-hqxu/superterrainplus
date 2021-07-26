set(STP_MAIN_ENGINE "SuperTerrain+")
# add SuperTerrain+ source codes to target
add_subdirectory(src)

# target setup
target_link_directories(${STP_MAIN_ENGINE}
PRIVATE ${CMAKE_SOURCE_DIR}/external/lib
)
target_include_directories(${STP_MAIN_ENGINE}
PRIVATE ${CMAKE_SOURCE_DIR}/external/include
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# link lib
target_link_libraries(${STP_MAIN_ENGINE}
PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc opengl32.lib glfw3.lib SglToolkit.lib SIMPLE.lib
)

# TODO: copy GLSL code to output