target_sources(${STP_MAIN_ENGINE} PRIVATE
STPCoreDefine.h
STPEngineInitialiser.h
STPOpenGL.h
STPSQLite.h
# /Environment
Environment/STPChunkSetting.h
Environment/STPConfiguration.h
Environment/STPErosionBrushGenerator.h
Environment/STPHeightfieldSetting.h
Environment/STPMeshSetting.h
Environment/STPRainDropSetting.cuh
Environment/STPSetting.hpp
# /GPGPU
GPGPU/STPHeightfieldKernel.cuh
GPGPU/STPRainDrop.cuh
GPGPU/STPRuntimeCompilable.h
# /Utility/Exception
Utility/Exception/STPAsyncGenerationError.h
Utility/Exception/STPBadNumericRange.h
Utility/Exception/STPCompilationError.h
Utility/Exception/STPCUDAError.h
Utility/Exception/STPDatabaseError.h
Utility/Exception/STPDeadThreadPool.h
Utility/Exception/STPInvalidArgument.h
Utility/Exception/STPInvalidEnvironment.h
Utility/Exception/STPMemoryError.h
Utility/Exception/STPSerialisationError.h
Utility/Exception/STPUnsupportedFunctionality.h
# /Utility/Memory
Utility/Memory/STPMemoryPool.h
Utility/Memory/STPSmartDeviceMemory.h
Utility/Memory/STPSmartDeviceMemory.tpp
Utility/Memory/STPSmartStream.h
# /Utility
Utility/STPDeviceErrorHandler.h
Utility/STPHashCombine.h
Utility/STPHashCombine.inl
Utility/STPThreadPool.h
Utility/STPThreadPool.inl
# /World/Chunk/FreeSlip
World/Chunk/FreeSlip/STPFreeSlipData.hpp
World/Chunk/FreeSlip/STPFreeSlipGenerator.h
World/Chunk/FreeSlip/STPFreeSlipLocation.hpp
World/Chunk/FreeSlip/STPFreeSlipManager.cuh
World/Chunk/FreeSlip/STPFreeSlipManager.inl
World/Chunk/FreeSlip/STPFreeSlipTextureBuffer.h
# /World/Chunk
World/Chunk/STPChunk.h
World/Chunk/STPChunkManager.h
World/Chunk/STPChunkProvider.h
World/Chunk/STPChunkStorage.h
World/Chunk/STPDiversityGenerator.hpp
World/Chunk/STPHeightfieldGenerator.h
# /World/Diversity
World/Diversity/STPBiomeDefine.h
World/Diversity/STPBiomeFactory.h
World/Diversity/STPLayer.h
World/Diversity/STPLayer.inl
World/Diversity/STPLayerCache.h
World/Diversity/STPLayerManager.h
World/Diversity/STPLayerManager.inl
World/Diversity/STPSeedMixer.h
# /World/Diversity/Texture
World/Diversity/Texture/STPTextureDatabase.h
World/Diversity/Texture/STPTextureDatabase.inl
World/Diversity/Texture/STPTextureFactory.h
World/Diversity/Texture/STPTextureInformation.hpp
World/Diversity/Texture/STPTextureType.hpp
)