target_sources(${STP_ALGORITHM_HOST_ENGINE} PRIVATE
STPPermutation.hpp
STPPermutationGenerator.h
STPSimplexNoiseSetting.h
)

target_sources(${STP_ALGORITHM_DEVICE_ENGINE} PRIVATE
Device/STPSimplexNoise.cuh
)