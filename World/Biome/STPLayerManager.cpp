#pragma once
#include "STPLayerManager.h"

using namespace SuperTerrainPlus::STPBiome;

STPLayer* STPLayerManager::start() {
	return this->Vertex.back().get();
}

size_t STPLayerManager::getLayerCount() const {
	return this->Vertex.size();
}