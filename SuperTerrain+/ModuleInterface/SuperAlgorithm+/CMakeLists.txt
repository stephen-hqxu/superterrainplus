target_sources(${STP_ALGORITHM_HOST_ENGINE} PRIVATE
STPAlgorithmDefine.h
STPPermutation.hpp
STPPermutationGenerator.h
STPSimplexNoiseSetting.h
STPSingleHistogram.hpp
STPSingleHistogramFilter.h
)

target_sources(${STP_ALGORITHM_DEVICE_ENGINE} PRIVATE
Device/STPKernelMath.cuh
Device/STPSimplexNoise.cuh
Device/STPSingleHistogramWrapper.cuh
Device/STPSingleHistogramWrapper.inl
)

set(STP_ALGORITHM_DEFINE_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/Include PARENT_SCOPE)