# host-side shared library
set(STP_ALGORITHM_HOST_ENGINE_SOURCE
STPAlgorithmEnvironment.cpp
STPINIParser.cpp
STPPermutationGenerator.cpp
STPSingleHistogramFilter.cpp
STPTextureDefinitionLanguage.cpp
)

add_library(${STP_ALGORITHM_HOST_ENGINE} SHARED
${STP_ALGORITHM_HOST_ENGINE_SOURCE}
${STP_ALGORITHM_HOST_ENGINE_INTERFACE}
)
# file filter for host library
# source
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}
PREFIX HostSource
FILES ${STP_ALGORITHM_HOST_ENGINE_SOURCE}
)
# header
source_group(
TREE ${STP_MODULE_INCLUDE}
FILES ${STP_ALGORITHM_HOST_ENGINE_INTERFACE}
)

set_target_properties(${STP_ALGORITHM_HOST_ENGINE} PROPERTIES DEFINE_SYMBOL SUPERALGORITHMPLUS_HOST_EXPORTS)

# device-side static library
set(STP_ALGORITHM_DEVICE_ENGINE_SOURCE
Device/STPSimplexNoise.cu
Device/STPTextureSplatRuleWrapper.cu
)

add_library(${STP_ALGORITHM_DEVICE_ENGINE} STATIC
${STP_ALGORITHM_DEVICE_ENGINE_SOURCE}
${STP_ALGORITHM_DEVICE_ENGINE_INTERFACE}
)
# file filter for device library
# source
source_group(
TREE ${CMAKE_CURRENT_SOURCE_DIR}/Device
PREFIX DeviceSource
FILES ${STP_ALGORITHM_DEVICE_ENGINE_SOURCE}
)
# header
source_group(
TREE ${STP_MODULE_INCLUDE}
FILES ${STP_ALGORITHM_DEVICE_ENGINE_INTERFACE}
)