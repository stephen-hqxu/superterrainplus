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
set(CMAKE_CUDA_ARCHITECTURES ${STP_CUDA_ARCH}) # -arch
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON) # -dlink=true

if(MSVC)
	
endif()

# output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_SHARED_LIBRARY_PREFIX "")