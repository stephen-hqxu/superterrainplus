target_sources(${STP_MAIN_ENGINE} PRIVATE
STPCoreDefine.h
STPEngineInitialiser.h
# /Environment
Environment/STPChunkSetting.h
Environment/STPConfiguration.h
Environment/STPErosionBrushGenerator.h
Environment/STPHeightfieldSetting.h
Environment/STPMeshSetting.h
Environment/STPRainDropSetting.cuh
Environment/STPSetting.hpp
# /GPGPU/FreeSlip
GPGPU/FreeSlip/STPFreeSlipData.hpp
GPGPU/FreeSlip/STPFreeSlipGenerator.cuh
GPGPU/FreeSlip/STPFreeSlipLocation.hpp
GPGPU/FreeSlip/STPFreeSlipManager.cuh
GPGPU/FreeSlip/STPFreeSlipManager.inl
GPGPU/FreeSlip/STPFreeSlipTextureBuffer.h
# /GPGPU
GPGPU/STPDiversityGenerator.hpp
GPGPU/STPDiversityGeneratorRTC.h
GPGPU/STPHeightfieldGenerator.cuh
GPGPU/STPRainDrop.cuh
# /Utility/Exception
Utility/Exception/STPAsyncGenerationError.h
Utility/Exception/STPBadNumericRange.h
Utility/Exception/STPCompilationError.h
Utility/Exception/STPCUDAError.h
Utility/Exception/STPDeadThreadPool.h
Utility/Exception/STPInvalidArgument.h
Utility/Exception/STPInvalidEnvironment.h
Utility/Exception/STPMemoryError.h
Utility/Exception/STPSerialisationError.h
Utility/Exception/STPUnsupportedFunctionality.h
# /Utility
Utility/STPDeviceErrorHandler.h
Utility/STPMemoryPool.h
Utility/STPSmartDeviceMemory.h
Utility/STPSmartDeviceMemory.tpp
Utility/STPSmartStream.h
Utility/STPThreadPool.h
Utility/STPThreadPool.inl
# /World/Chunk
World/Chunk/STPChunk.h
World/Chunk/STPChunkManager.h
World/Chunk/STPChunkProvider.h
World/Chunk/STPChunkStorage.h
# /World/Diversity
World/Diversity/STPBiomeDefine.h
World/Diversity/STPBiomeFactory.h
World/Diversity/STPLayer.h
World/Diversity/STPLayer.inl
World/Diversity/STPLayerCache.h
World/Diversity/STPLayerManager.h
World/Diversity/STPLayerManager.inl
World/Diversity/STPSeedMixer.h
World/Diversity/STPTextureDatabase.h
)