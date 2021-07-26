#pragma once
#include <World/Biome/Layers/STPAllLayers.h>

//All layers are for demo purposes, developers are free to implement their own version with the interface

//Include all layers in this files, do not include each layer individually
//Note that all layers are for demo, in real application programmers can have their own implementation

//Composite layer
#include <World/Biome/Layers/STPCrossLayer.h>
#include <World/Biome/Layers/STPXCrossLayer.h>
#include <World/Biome/Layers/STPVoronoiLayer.h>

//Land layer
#include <World/Biome/Layers/STPContinentLayer.h>
#include <World/Biome/Layers/STPLandLayer.h>
#include <World/Biome/Layers/STPIslandLayer.h>
#include <World/Biome/Layers/STPBaseBiomeLayer.h>
#include <World/Biome/Layers/STPNoiseLayer.h>
#include <World/Biome/Layers/STPHillsLayer.h>

//Shore layer
#include <World/Biome/Layers/STPEdgeBiomeLayer.h>
#include <World/Biome/Layers/STPEaseEdgeLayer.h>

//Temperature layer
#include <World/Biome/Layers/STPClimateLayer.h>
#include <World/Biome/Layers/STPOceanTemperatureLayer.h>

//Water layer
#include <World/Biome/Layers/STPDeepOceanLayer.h>
#include <World/Biome/Layers/STPNoiseToRiverLayer.h>
#include <World/Biome/Layers/STPRiverErodeLayer.h>
#include <World/Biome/Layers/STPRiverMixLayer.h>

//Scale layer
#include <World/Biome/Layers/STPScaleLayer.h>
#include <World/Biome/Layers/STPSmoothScaleLayer.h>

using namespace STPDemo;

using glm::uvec2;

//The size of individual cache
constexpr size_t Cachesize = 2048ull;

STPLayerChainBuilder::STPLayerChainBuilder(uvec2 dimension, Seed global) : GlobalSeed(global), STPBiomeFactory(dimension) {

}

SuperTerrainPlus::STPDiversity::STPLayerManager* STPLayerChainBuilder::supply() const {
	using namespace SuperTerrainPlus::STPDiversity;
	//create a new manager, don't worry about deletion because our engine will manage it pretty well
	STPLayerManager* chain = new STPLayerManager();
	STPLayer* base;

	//building layer chain
	//we use a hand-typed random salt
	//base biome
	//4096
	base = chain->insert<STPContinentLayer, Cachesize>(this->GlobalSeed, 23457829ull);
	//2048
	base = chain->insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 875944ull, STPScaleLayer::STPScaleType::FUZZY, base);
	base = chain->insert<STPLandLayer, Cachesize>(this->GlobalSeed, 5748329ull, base);
	//1024
	base = chain->insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 8947358941ull, STPScaleLayer::STPScaleType::NORMAL, base);
	base = chain->insert<STPLandLayer, Cachesize>(this->GlobalSeed, 361249673ull, base);
	base = chain->insert<STPLandLayer, Cachesize>(this->GlobalSeed, 8769575ull, base);
	base = chain->insert<STPLandLayer, Cachesize>(this->GlobalSeed, 43562783426564ull, base);
	base = chain->insert<STPIslandLayer, Cachesize>(this->GlobalSeed, 74368ull, base);

	//debug for speedy generation
	base = chain->insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 1ull, STPScaleLayer::STPScaleType::NORMAL, base);
	base = chain->insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 2ull, STPScaleLayer::STPScaleType::NORMAL, base);
	base = chain->insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 3ull, STPScaleLayer::STPScaleType::NORMAL, base);

	return chain;
}