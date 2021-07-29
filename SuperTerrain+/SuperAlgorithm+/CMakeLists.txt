add_library(${STP_ALGORITHM_ENGINE} STATIC
STPSimplexNoiseSetting.cpp

STPPermutationGenerator.cu
STPSimplexNoise.cu
STPSimplexNoiseRTC.cu
)

# link CUDA library
target_link_libraries(${STP_ALGORITHM_ENGINE}
PUBLIC CUDA::cudart
)