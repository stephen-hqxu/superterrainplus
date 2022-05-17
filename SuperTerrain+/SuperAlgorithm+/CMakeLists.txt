# host-side shared library
add_library(${STP_ALGORITHM_HOST_ENGINE} SHARED
STPAlgorithmEnvironment.cpp
STPINIParser.cpp
STPPermutationGenerator.cpp
STPSingleHistogramFilter.cpp
STPTextureDefinitionLanguage.cpp
)
set_target_properties(${STP_ALGORITHM_HOST_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERALGORITHMPLUS_HOST_EXPORTS)

# device-side static library
add_library(${STP_ALGORITHM_DEVICE_ENGINE} STATIC
Device/STPKernelMath.cu
Device/STPSimplexNoise.cu
Device/STPSingleHistogramWrapper.cu
Device/STPTextureSplatRuleWrapper.cu
)