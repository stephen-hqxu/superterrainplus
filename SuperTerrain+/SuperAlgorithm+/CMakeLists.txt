# host-side shared library
add_library(${STP_ALGORITHM_HOST_ENGINE} SHARED
STPPermutationGenerator.cpp
STPSimplexNoiseSetting.cpp
STPSingleHistogramFilter.cpp
)
set_target_properties(${STP_ALGORITHM_HOST_ENGINE} PROPERTIES DEFINE_SYMBOL STPALGORITHMPLUS_HOST_EXPORTS)

# device-side static library
add_library(${STP_ALGORITHM_DEVICE_ENGINE} STATIC
Device/STPKernelMath.cu
Device/STPSimplexNoise.cu
)