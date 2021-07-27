#pragma once
#include <World/Diversity/STPLayerManager.h>

using namespace SuperTerrainPlus::STPDiversity;

void STPLayerManager::STPLayerRecycler::operator()(STPLayer* ptr) const {
	delete ptr;
}

STPLayer* STPLayerManager::start() {
	return this->Vertex.back().get();
}

size_t STPLayerManager::getLayerCount() const {
	return this->Vertex.size();
}