add_executable(${STP_MAIN_ENGINE}
glad.c
# main engine src
STPBiomeFactory.cpp
STPChunk.cpp
STPChunkManager.cpp
STPChunkProvider.cpp
STPChunkStorage.cpp
STPDeviceErrorHandler.cpp
STPDiversityGeneratorRTC.cpp
STPLayer.cpp
STPLayerCache.cpp
STPLayerManager.cpp
STPSeedMixer.cpp
STPThreadPool.cpp

STPHeightfieldGenerator.cu
STPPermutationsGenerator.cu
STPRainDrop.cu
STPRainDropSettings.cu
STPSimplexNoise.cu
# demo program src, I will separate demo program into another target or project later
STPAllLayers.cpp
STPBiome.cpp
STPBiomefieldGenerator.cpp
STPProcedural2DINF.cpp
STPStart.cpp
STPTerrainParaLoader.cpp
STPTextureStorage.cpp
STPWorldManager.cpp

STPMultiHeightGenerator.cu
)