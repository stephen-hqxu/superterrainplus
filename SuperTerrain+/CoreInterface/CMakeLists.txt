target_sources(${STP_MAIN_ENGINE} PRIVATE
STPEngineInitialiser.h
# /Environment
Environment/STPChunkSetting.h
Environment/STPConfiguration.h
Environment/STPHeightfieldSetting.h
Environment/STPMeshSetting.h
Environment/STPRainDropSetting.h
Environment/STPSetting.hpp
# /GPGPU
GPGPU/STPDiversityGenerator.hpp
GPGPU/STPDiversityGeneratorRTC.h
GPGPU/STPHeightfieldGenerator.cuh
GPGPU/STPRainDrop.cuh
# /Utility
Utility/STPMemoryPool.hpp
Utility/STPSerialisationException.hpp
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
)