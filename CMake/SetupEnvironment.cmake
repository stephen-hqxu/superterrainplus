# import GLM and OpenGL
find_package(glm REQUIRED CONFIG)
find_package(OpenGL REQUIRED)

# minimum supported CUDA versions
set(STP_CUDA_MIN_VERSION 11.3)

# setup CUDA
include(FindCUDAToolkit)
# error handling
if(NOT ${CUDAToolkit_FOUND})
	message(FATAL_ERROR "CUDA compiler is not found.")
endif()
if(${CUDAToolkit_VERSION} VERSION_LESS ${STP_CUDA_MIN_VERSION})
	message(FATAL_ERROR "CUDA version is incompatible.")
endif()

# setup default compiler options
set(CMAKE_CXX_STANDARD 17) # std=c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # -rdc=true
set(CMAKE_CUDA_RUNTIME_LIBRARY ${STP_CUDA_RUNTIME_LIBRARY}) # -cudart=...
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}-real ${CMAKE_CUDA_ARCHITECTURES}-virtual) # -arch
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")# --extended-lambda
if(${STP_CUDA_VERBOSE_PTX})
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# --ptxas-options=-v
endif()
# turn on device debug information generation on debug mode
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -G")

if(MSVC)
	if(${STP_USE_AVX2})
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")# /arch:AVX2
	endif()
endif()

# output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_SHARED_LIBRARY_PREFIX "")