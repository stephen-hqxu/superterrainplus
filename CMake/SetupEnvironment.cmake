# import installed libraries
find_package(glm REQUIRED CONFIG)
# import system libraries
find_package(OpenGL REQUIRED)
find_package(SQLite3 3.30 REQUIRED)
find_package(CUDAToolkit 11.7 REQUIRED)
find_package(OptiX 7.3 REQUIRED)

# setup default compiler options
set(CMAKE_CXX_STANDARD 17) # -std=c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# compiler and linker warning, and other flags
if(MSVC)
	# MSVC legacy lambda breaks sometimes if we define a super complex lambda (like super nesting)
	# so we want the standard-conforming lambda
	# TODO: this can be removed when the project is ported to C++ 20
	# we also need the C++ version macro for some external libraries to detect the standard correctly
	set(STP_MSVC_FLAG "/arch:AVX2 /Zc:lambda /Zc:__cplusplus")
	# disable warning about using stl in a dll project
	set(STP_MSVC_FLAG "${STP_MSVC_FLAG} /wd4251 /wd4275")
	set(STP_MSVC_CUDA_FLAG "-err-no -diag-suppress 1388,1394")

	if(STP_ENABLE_WARNING)
		set(STP_MSVC_FLAG "${STP_MSVC_FLAG} /W4")
	endif()

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STP_MSVC_FLAG}")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${STP_MSVC_CUDA_FLAG} -Xcompiler \"${STP_MSVC_FLAG}\"")
	unset(STP_MSVC_CUDA_FLAG)
	unset(STP_MSVC_FLAG)
else()
	set(STP_COMPILER_FLAG "-mavx2 -mfma")

	if(STP_ENABLE_WARNING)
		set(STP_COMPILER_FLAG "${STP_COMPILER_FLAG} -Wall -Wextra -pedantic")
	endif()

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STP_COMPILER_FLAG}")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${STP_COMPILER_FLAG}\"")
	unset(STP_COMPILER_FLAG)
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