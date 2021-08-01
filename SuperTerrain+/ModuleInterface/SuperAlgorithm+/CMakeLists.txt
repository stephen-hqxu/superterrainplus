target_sources(${STP_ALGORITHM_HOST_ENGINE} PRIVATE
STPPermutation.hpp
STPPermutationGenerator.h
STPSimplexNoiseSetting.h
)

target_sources(${STP_ALGORITHM_DEVICE_ENGINE} PRIVATE
Device/STPSimplexNoise.cuh
)

set(STP_ALGORITHM_DEFINE_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/Include PARENT_SCOPE)