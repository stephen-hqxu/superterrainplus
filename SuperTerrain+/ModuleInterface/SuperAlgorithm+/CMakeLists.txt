target_sources(${STP_ALGORITHM_HOST_ENGINE} PRIVATE
STPAlgorithmDefine.h
STPPermutation.hpp
STPPermutationGenerator.h
STPSimplexNoiseSetting.h
STPSingleHistogram.hpp
STPSingleHistogramFilter.h
# /Parser/INI
Parser/INI/STPINIBasicString.h
Parser/INI/STPINIBasicString.inl
Parser/INI/STPINIReader.h
Parser/INI/STPINISection.hpp
Parser/INI/STPINIStorage.hpp
Parser/INI/STPINIWriter.h
# /Parser
Parser/STPTextureDefinitionLanguage.h
)

target_sources(${STP_ALGORITHM_DEVICE_ENGINE} PRIVATE
Device/STPSimplexNoise.cuh
Device/STPSingleHistogramWrapper.cuh
Device/STPSingleHistogramWrapper.inl
Device/STPTextureSplatRuleWrapper.cuh
)

# config file for algorithm engine
setupConfigTemplate(${STP_ALGORITHM_ENGINE} STPAlgorithmDeviceInfo.h.in Include/STPAlgorithmDeviceInfo.h STPAlgorithmDeviceInfo.h)