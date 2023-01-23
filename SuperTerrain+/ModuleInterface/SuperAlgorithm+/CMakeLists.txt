# algorithm host engine
set(STP_ALGORITHM_HOST_ENGINE_INTERFACE
STPAlgorithmDefine.h
STPPermutation.hpp
STPPermutationGenerator.h
STPSimplexNoiseSetting.h
STPSingleHistogram.hpp
STPSingleHistogramFilter.h
# /Parser/Framework
Parser/Framework/STPBasicStringAdaptor.h
Parser/Framework/STPBasicStringAdaptor.inl
Parser/Framework/STPLexer.h
Parser/Framework/STPLexer.inl
Parser/Framework/STPRegularLanguage.h
Parser/Framework/STPRegularLanguage.inl
# /Parser
Parser/STPINIData.hpp
Parser/STPINIParser.h
Parser/STPTextureDefinitionLanguage.h
)

list(TRANSFORM STP_ALGORITHM_HOST_ENGINE_INTERFACE PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set(STP_ALGORITHM_HOST_ENGINE_INTERFACE ${STP_ALGORITHM_HOST_ENGINE_INTERFACE} PARENT_SCOPE)

# algorithm device engine
set(STP_ALGORITHM_DEVICE_ENGINE_INTERFACE
Device/STPSimplexNoise.cuh
Device/STPSingleHistogramWrapper.cuh
Device/STPSingleHistogramWrapper.inl
Device/STPTextureSplatRuleWrapper.cuh
)

list(TRANSFORM STP_ALGORITHM_DEVICE_ENGINE_INTERFACE PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set(STP_ALGORITHM_DEVICE_ENGINE_INTERFACE ${STP_ALGORITHM_DEVICE_ENGINE_INTERFACE} PARENT_SCOPE)

# config file for algorithm engine
setupConfigTemplate(${STP_ALGORITHM_ENGINE} STPAlgorithmDeviceInfo.h.in Include/STPAlgorithmDeviceInfo.h STPAlgorithmDeviceInfo.h)