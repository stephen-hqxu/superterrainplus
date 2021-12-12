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
# /Exception
Exception/STPAsyncGenerationError.h
Exception/STPBadNumericRange.h
Exception/STPCompilationError.h
Exception/STPCUDAError.h
Exception/STPDatabaseError.h
Exception/STPDeadThreadPool.h
Exception/STPGLError.h
Exception/STPInvalidArgument.h
Exception/STPInvalidEnvironment.h
Exception/STPInvalidSyntax.h
Exception/STPMemoryError.h
Exception/STPSerialisationError.h
Exception/STPUnsupportedFunctionality.h
# /GPGPU
GPGPU/STPHeightfieldKernel.cuh
GPGPU/STPRainDrop.cuh
GPGPU/STPRuntimeCompilable.h
# /Utility/Memory
Utility/Memory/STPMemoryPool.h
Utility/Memory/STPSmartDeviceMemory.h
Utility/Memory/STPSmartDeviceMemory.tpp
Utility/Memory/STPSmartStream.h
# /Utility
Utility/STPDeviceErrorHandler.h
Utility/STPFile.h
Utility/STPFile.inl
Utility/STPHashCombine.h
Utility/STPHashCombine.inl
Utility/STPNullablePrimitive.h
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
World/Diversity/Texture/STPTextureDefinitionLanguage.h
World/Diversity/Texture/STPTextureFactory.h
World/Diversity/Texture/STPTextureInformation.hpp
World/Diversity/Texture/STPTextureType.hpp
# /World
World/STPWorldPipeline.h
)