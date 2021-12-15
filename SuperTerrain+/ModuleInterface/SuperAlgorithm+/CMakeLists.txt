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
Device/STPKernelMath.inl
Device/STPSimplexNoise.cuh
Device/STPSingleHistogramWrapper.cuh
Device/STPSingleHistogramWrapper.inl
Device/STPTextureSplatRuleWrapper.cuh
)

# config file for algorithm engine
# tell the static library location so CUDA runtime compiler can find it easily (if developer wants to)
# algorithm device info contains path to the static library, we need to output that based on configuration
setupConfigTemplate(${STP_ALGORITHM_ENGINE} STPAlgorithmDeviceInfo.h.in Include/STPAlgorithmDeviceInfoRaw.h STPAlgorithmDeviceInfo.h)