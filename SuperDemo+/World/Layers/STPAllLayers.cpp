#include "STPAllLayers.h"

//All layers are for demo purposes, developers are free to implement their own version with the interface

//Include all layers in this files, do not include each layer individually
//Note that all layers are for demo, in real application programmers can have their own implementation

//Composite layer
#include "STPCrossLayer.h"
#include "STPXCrossLayer.h"
#include "STPVoronoiLayer.h"

//Land layer
#include "STPContinentLayer.h"
#include "STPLandLayer.h"
#include "STPIslandLayer.h"
#include "STPBaseBiomeLayer.h"
#include "STPNoiseLayer.h"
#include "STPHillsLayer.h"

//Shore layer
#include "STPEdgeBiomeLayer.h"
#include "STPEaseEdgeLayer.h"

//Temperature layer
#include "STPClimateLayer.h"
#include "STPOceanTemperatureLayer.h"

//Water layer
#include "STPDeepOceanLayer.h"
#include "STPNoiseToRiverLayer.h"
#include "STPRiverErodeLayer.h"
#include "STPRiverMixLayer.h"

//Scale layer
#include "STPScaleLayer.h"
#include "STPSmoothScaleLayer.h"

using namespace STPDemo;
using namespace SuperTerrainPlus::STPDiversity;

using glm::uvec2;

//The size of individual cache
static constexpr size_t Cachesize = 2048ull;

STPLayerChainBuilder::STPLayerChainBuilder(uvec2 dimension, Seed global) : GlobalSeed(global), STPBiomeFactory(dimension) {

}

STPLayerManager STPLayerChainBuilder::supply() const {
	//create a new manager, don't worry about deletion because our engine will manage it pretty well
	STPLayerManager chain;
	STPLayer* base;

	//building layer chain
	//we use a hand-typed random salt
	//base biome
	//4096
	base = chain.insert<STPContinentLayer, Cachesize>(this->GlobalSeed, 23457829ull);
	//2048
	base = chain.insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 875944ull, STPScaleLayer::STPScaleType::FUZZY, base);
	base = chain.insert<STPLandLayer, Cachesize>(this->GlobalSeed, 5748329ull, base);
	//1024
	base = chain.insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 8947358941ull, STPScaleLayer::STPScaleType::NORMAL, base);
	base = chain.insert<STPLandLayer, Cachesize>(this->GlobalSeed, 361249673ull, base);
	base = chain.insert<STPLandLayer, Cachesize>(this->GlobalSeed, 8769575ull, base);
	base = chain.insert<STPLandLayer, Cachesize>(this->GlobalSeed, 43562783426564ull, base);
	base = chain.insert<STPIslandLayer, Cachesize>(this->GlobalSeed, 74368ull, base);

	//debug for speedy generation
	base = chain.insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 1ull, STPScaleLayer::STPScaleType::NORMAL, base);
	base = chain.insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 2ull, STPScaleLayer::STPScaleType::NORMAL, base);
	base = chain.insert<STPScaleLayer, Cachesize>(this->GlobalSeed, 3ull, STPScaleLayer::STPScaleType::NORMAL, base);
	base = chain.insert<STPVoronoiLayer, Cachesize>(this->GlobalSeed, 4ull, false, base);
	base = chain.insert<STPVoronoiLayer, Cachesize>(this->GlobalSeed, 5ull, false, base);
	//the last layer can be uncached because each pixel is only referenced once
	base = chain.insert<STPVoronoiLayer>(this->GlobalSeed, 6ull, false, base);

	return chain;
}