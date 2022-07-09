# import installed libraries
find_package(glm REQUIRED CONFIG)
# import system libraries
find_package(OpenGL REQUIRED)
find_package(SQLite3 3.30 REQUIRED)
find_package(CUDAToolkit 11.3 REQUIRED)
find_package(OptiX 7 REQUIRED)

# setup default compiler options
set(CMAKE_CXX_STANDARD 17) # -std=c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# compiler and linker warning, and other flags
if(MSVC)
	set(STP_MSVC_WARNING "/wd4251 /wd4275") # disable warning about using stl in a dll project
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STP_MSVC_WARNING}")
	if(STP_ENABLE_WARNING)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
	endif()

	if(STP_USE_AVX2)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
	endif()
else()
	if(STP_ENABLE_WARNING)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
	endif()

	if(STP_USE_AVX2)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
	endif()
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # -rdc=true
set(CMAKE_CUDA_RUNTIME_LIBRARY ${STP_CUDA_RUNTIME_LIBRARY}) # -cudart=...
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda") # --extended-lambda
if(STP_CUDA_VERBOSE_PTX)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v") # --ptxas-options=-v
endif()
# turn on device debug information generation on debug mode
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -G")

# output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)