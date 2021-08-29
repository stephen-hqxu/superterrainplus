# CUDA
set(STP_CUDA_RUNTIME_LIBRARY "Static" CACHE STRING "Value for nvcc compiler option -cudart=")
option(STP_CUDA_VERBOSE_PTX "Use nvcc compiler flag --ptxas-options=-v for detailed PTX output" OFF)
# Others
option(STP_USE_AVX2 "Enable AVX2 instruction sets on vector operations" ON)
option(STP_BUILD_DEMO "Build SuperDemo+, a demo program for SuperTerrain+ engine" ON)
option(STP_BUILD_TEST "Build SuperTest+, a unit test program" OFF)