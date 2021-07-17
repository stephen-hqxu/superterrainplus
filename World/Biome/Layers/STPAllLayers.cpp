#pragma once
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

STPLayerChainBuilder::STPLayerChainBuilder(Seed global) : GlobalSeed(global) {

}

SuperTerrainPlus::STPDiversity::STPLayerManager* STPLayerChainBuilder::operator()() const {
	using namespace SuperTerrainPlus::STPDiversity;
	//create a new manager, don't worry about deletion because our engine will manage it pretty well
	STPLayerManager* chain = new STPLayerManager();

	//building layer chain
	//we use a hand-made random salt

	return chain;
}