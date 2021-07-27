add_library(${STP_ALGORITHM_ENGINE} STATIC
STPPermutationsGenerator.cu
STPSimplexNoise.cu
)

# link CUDA library
target_link_libraries(${STP_ALGORITHM_ENGINE}
PUBLIC CUDA::cudart
)