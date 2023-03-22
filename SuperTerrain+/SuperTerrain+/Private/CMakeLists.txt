target_sources(${STP_MAIN_ENGINE} PRIVATE
STPEngineInitialiser.cpp
STPEnvironment.cpp
STPException.cpp
# /GPGPU
GPGPU/STPDeviceRuntimeBinary.cpp
GPGPU/STPDeviceRuntimeProgram.cpp

GPGPU/STPHeightfieldKernel.cu
GPGPU/STPRainDrop.cu
# /Utility
Utility/STPFile.cpp
Utility/STPSmartDeviceObject.cpp
Utility/STPThreadPool.cpp
# /World/Chunk
World/Chunk/STPChunk.cpp
World/Chunk/STPHeightfieldGenerator.cpp
World/Chunk/STPNearestNeighbourTextureBuffer.cpp
# /World/Diversity
World/Diversity/STPBiomeFactory.cpp
World/Diversity/STPLayer.cpp
World/Diversity/STPTextureDatabase.cpp
World/Diversity/STPTextureFactory.cpp
# /World
World/STPWorldPipeline.cpp
)