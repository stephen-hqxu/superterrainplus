target_sources(${STP_MAIN_ENGINE} PRIVATE
STPCoreDefine.h
STPEngineInitialiser.h
STPOpenGL.h
STPSQLite.h
# /Environment
Environment/STPChunkSetting.h
Environment/STPHeightfieldSetting.h
Environment/STPRainDropSetting.h
# /Exception/API
Exception/API/STPCUDAError.h
Exception/API/STPGLError.h
Exception/API/STPSQLError.h
# /Exception
Exception/STPFundamentalException.h
Exception/STPInsufficientMemory.h
Exception/STPInvalidEnum.h
Exception/STPInvalidEnvironment.h
Exception/STPIOException.h
Exception/STPListenerError.h
Exception/STPNumericDomainError.h
Exception/STPParserError.h
Exception/STPUnimplementedFeature.h
Exception/STPUnsupportedSystem.h
Exception/STPValidationFailed.h
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
Utility/Memory/STPObjectPool.h
Utility/Memory/STPObjectPool.inl
Utility/Memory/STPSmartDeviceMemory.h
Utility/Memory/STPSmartDeviceMemory.inl
Utility/Memory/STPSmartDeviceObject.h
# /Utility
Utility/STPDatabaseErrorHandler.hpp
Utility/STPDeviceErrorHandler.hpp
Utility/STPFile.h
Utility/STPGenericErrorHandlerBlueprint.hpp
Utility/STPHashCombine.h
Utility/STPHashCombine.inl
Utility/STPNullablePrimitive.h
Utility/STPNullablePrimitive.inl
Utility/STPStringUtility.h
Utility/STPStringUtility.inl
Utility/STPThreadPool.h
Utility/STPThreadPool.inl
# /World/Chunk
World/Chunk/STPChunk.h
World/Chunk/STPDiversityGenerator.hpp
World/Chunk/STPErosionBrush.hpp
World/Chunk/STPHeightfieldGenerator.h
World/Chunk/STPNearestNeighbourInformation.hpp
World/Chunk/STPNearestNeighbourTextureBuffer.h
# /World/Diversity
World/Diversity/STPBiomeDefine.h
World/Diversity/STPBiomeFactory.h
World/Diversity/STPLayer.h
World/Diversity/STPLayer.inl
# /World/Diversity/Texture
World/Diversity/Texture/STPTextureDatabase.h
World/Diversity/Texture/STPTextureFactory.h
World/Diversity/Texture/STPTextureInformation.hpp
World/Diversity/Texture/STPTextureType.hpp
# /World
World/STPWorldPipeline.h
)