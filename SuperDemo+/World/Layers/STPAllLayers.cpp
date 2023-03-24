#include "STPAllLayers.h"

//Biome
#include <SuperTerrain+/World/Diversity/STPLayer.h>
#include "../Biomes/STPBiomeRegistry.h"

using namespace SuperTerrainPlus::STPDiversity;
namespace Reg = STPDemo::STPBiomeRegistry;

//All layers are for demo purposes, developers are free to implement their own version with the interface

//Include all layers in this files, do not include each layer individually; they all have internal linkage.
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

/* --------------------------------- */

using namespace STPDemo;

//The size of individual cache
static constexpr size_t CacheSize = 2048u;

#define LAYER_COMMON CacheSize, global

struct STPLayerChainBuilder::STPLayerPipeline {
public:

	//for reduced verbosity, we use obfuscated naming, plus naming doesn't help much here
	//base biome
	//4096
	STPContinentLayer A0;
	//2048
	STPScaleLayer B0;
	STPLandLayer B1;
	//1024
	STPScaleLayer C0;
	STPLandLayer C1, C2, C3;
	STPIslandLayer C4;

	//debug for speedy generation
	STPScaleLayer D0, D1, D2;
	STPVoronoiLayer D3, D4, D5;

	//The layer where generation should start;
	//remember to change it whenever the layer pipeline changes.
	STPLayer& Root = this->D5;

	/**
	 * @brief Create a new layer pipeline.
	 * @param global The global seed.
	*/
	STPLayerPipeline(const Seed global) :
		A0(LAYER_COMMON, 23457829ull),
		
		B0(LAYER_COMMON, 875944ull, STPScaleLayer::STPScaleType::FUZZY, this->A0),
		B1(LAYER_COMMON, 5748329ull, this->B0),

		C0(LAYER_COMMON, 8947358941ull, STPScaleLayer::STPScaleType::NORMAL, this->B1),
		C1(LAYER_COMMON, 361249673ull, this->C0),
		C2(LAYER_COMMON, 8769575ull, this->C1),
		C3(LAYER_COMMON, 43562783426564ull, this->C2),
		C4(LAYER_COMMON, 74368ull, this->C3),

		D0(LAYER_COMMON, 1ull, STPScaleLayer::STPScaleType::NORMAL, this->C4),
		D1(LAYER_COMMON, 2ull, STPScaleLayer::STPScaleType::NORMAL, this->D0),
		D2(LAYER_COMMON, 3ull, STPScaleLayer::STPScaleType::NORMAL, this->D1),
		D3(LAYER_COMMON, 4ull, false, this->D2),
		D4(LAYER_COMMON, 5ull, false, this->D3),
		//the last layer can be uncached because each pixel is only referenced once
		D5(0ull, global, 6ull, false, this->D4) {

	}

	~STPLayerPipeline() = default;

};

STPLayerChainBuilder::STPLayerChainBuilder(const glm::uvec2 dimension, const Seed global) : STPBiomeFactory(dimension), GlobalSeed(global) {

}

STPDemo::STPLayerChainBuilder::~STPLayerChainBuilder() = default;

STPLayer& STPLayerChainBuilder::supply() {
	return this->LayerStructureStorage.emplace_back(this->GlobalSeed).Root;
}