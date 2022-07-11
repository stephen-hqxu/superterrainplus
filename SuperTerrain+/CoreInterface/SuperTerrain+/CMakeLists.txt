set(STP_MAIN_ENGINE_INTERFACE
STPCoreDefine.h
STPEngineInitialiser.h
STPOpenGL.h
STPSQLite.h
# /Environment
Environment/STPChunkSetting.h
Environment/STPConfiguration.h
Environment/STPHeightfieldSetting.h
Environment/STPRainDropSetting.h
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
GPGPU/STPDeviceRuntimeBinary.h
GPGPU/STPDeviceRuntimeProgram.h
GPGPU/STPHeightfieldKernel.cuh
GPGPU/STPRainDrop.cuh
# /Utility/Algebra
Utility/Algebra/STPAlgebraCommon.h
Utility/Algebra/STPAlgebraCommon.inl
Utility/Algebra/STPMatrix4x4d.h
Utility/Algebra/STPMatrix4x4d.inl
Utility/Algebra/STPVector4d.h
Utility/Algebra/STPVector4d.inl
# /Utility/Memory
Utility/Memory/STPMemoryPool.h
Utility/Memory/STPObjectPool.h
Utility/Memory/STPObjectPool.inl
Utility/Memory/STPSmartDeviceMemory.h
Utility/Memory/STPSmartDeviceMemory.inl
Utility/Memory/STPSmartEvent.h
Utility/Memory/STPSmartStream.h
# /Utility
Utility/STPDeviceErrorHandler.h
Utility/STPDeviceErrorHandlerBlueprint.hpp
Utility/STPFile.h
Utility/STPFile.inl
Utility/STPHashCombine.h
Utility/STPHashCombine.inl
Utility/STPNullablePrimitive.h
Utility/STPNullablePrimitive.inl
Utility/STPThreadPool.h
Utility/STPThreadPool.inl
# /World/Chunk
World/Chunk/STPChunk.h
World/Chunk/STPDiversityGenerator.hpp
World/Chunk/STPErosionBrush.hpp
World/Chunk/STPErosionBrushGenerator.h
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
World/Diversity/Texture/STPTextureFactory.h
World/Diversity/Texture/STPTextureInformation.hpp
World/Diversity/Texture/STPTextureType.hpp
# /World
World/STPWorldPipeline.h
)

# prefix main engine header path
list(TRANSFORM STP_MAIN_ENGINE_INTERFACE PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
set(STP_MAIN_ENGINE_INTERFACE ${STP_MAIN_ENGINE_INTERFACE} PARENT_SCOPE)