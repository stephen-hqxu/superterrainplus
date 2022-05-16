# CUDA
set(STP_CUDA_RUNTIME_LIBRARY "Shared" CACHE STRING "Value for nvcc compiler option -cudart=")
# Others
option(STP_USE_AVX2 "Enable AVX2 instruction sets on vector operations" ON)
option(STP_BUILD_DEMO "Build SuperDemo+, a demo program for SuperTerrain+ engine" ON)
option(STP_DEVELOPMENT_BUILD "Enable development mode" OFF)

include(CMakeDependentOption)
# dependent options
cmake_dependent_option(STP_CUDA_VERBOSE_PTX "Use nvcc compiler flag --ptxas-options=-v for detailed PTX output" ON "STP_DEVELOPMENT_BUILD" OFF)
cmake_dependent_option(STP_BUILD_TEST "Build SuperTest+, a unit test program" ON "STP_DEVELOPMENT_BUILD" OFF)
cmake_dependent_option(STP_ENABLE_WARNING "Enable all compiler warnings for development testing" ON "STP_DEVELOPMENT_BUILD" OFF)