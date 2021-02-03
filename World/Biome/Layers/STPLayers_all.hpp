#pragma once
#ifndef _STP_LAYERS_ALL_HPP_
#define _STP_LAYERS_ALL_HPP_

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

#endif//_STP_LAYERS_ALL_HPP_