#pragma once
#include "STPLayerManager.h"

using namespace SuperTerrainPlus::STPDiversity;

STPLayer* STPLayerManager::start() {
	return this->Vertex.back().get();
}

size_t STPLayerManager::getLayerCount() const {
	return this->Vertex.size();
}