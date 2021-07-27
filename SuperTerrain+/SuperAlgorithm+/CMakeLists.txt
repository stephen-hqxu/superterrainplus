add_library(${STP_ALGORITHM_ENGINE} STATIC
STPPermutationGenerator.cu
STPSimplexNoise.cu
)

# link CUDA library
target_link_libraries(${STP_ALGORITHM_ENGINE}
PUBLIC CUDA::cudart
)