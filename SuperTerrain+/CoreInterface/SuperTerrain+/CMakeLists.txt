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
# /Utility/Algebra
Utility/Algebra/STPMatrix4x4d.h
Utility/Algebra/STPVector4d.h
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
# /World/Chunk
World/Chunk/STPChunk.h
World/Chunk/STPDiversityGenerator.hpp
World/Chunk/STPFreeSlipInformation.hpp
World/Chunk/STPFreeSlipTextureBuffer.h
World/Chunk/STPHeightfieldGenerator.h
# /World/Diversity
World/Diversity/STPBiomeDefine.h
World/Diversity/STPBiomeFactory.h
World/Diversity/STPLayer.h
World/Diversity/STPLayer.inl
World/Diversity/STPLayerCache.h
World/Diversity/STPLayerManager.h
World/Diversity/STPLayerManager.inl
# /World/Diversity/Texture
World/Diversity/Texture/STPTextureDatabase.h
World/Diversity/Texture/STPTextureDefinitionLanguage.h
World/Diversity/Texture/STPTextureFactory.h
World/Diversity/Texture/STPTextureInformation.hpp
World/Diversity/Texture/STPTextureType.hpp
# /World
World/STPWorldPipeline.h
)