#pragma once
#include <World/Diversity/STPLayerManager.h>

using namespace SuperTerrainPlus::STPDiversity;

void STPLayerManager::recycleLayer(STPLayer* ptr) {
	delete ptr;
}

STPLayer* STPLayerManager::start() {
	return this->Vertex.back().get();
}

size_t STPLayerManager::getLayerCount() const {
	return this->Vertex.size();
}