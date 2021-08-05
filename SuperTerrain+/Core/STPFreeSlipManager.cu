#pragma once
#include <GPGPU/STPFreeSlipManager.cuh>

using namespace SuperTerrainPlus::STPCompute;

__host__ STPFreeSlipManager::STPFreeSlipManager(float* heightmap, const unsigned int* index, uint2 range, uint2 mapSize)
	: Dimension(mapSize), FreeSlipChunk(range), FreeSlipRange(make_uint2(range.x* mapSize.x, range.y* mapSize.y)) {
	this->Heightmap = heightmap;
	this->Index = index;
}

__host__ STPFreeSlipManager::~STPFreeSlipManager() {

}

__device__ float& STPFreeSlipManager::operator[](unsigned int global) {
	return const_cast<float&>(const_cast<const STPFreeSlipManager*>(this)->operator[](global));
}

__device__ const float& STPFreeSlipManager::operator[](unsigned int global) const {
	return this->Heightmap[this->operator()(global)];
}

__device__ unsigned int STPFreeSlipManager::operator()(unsigned int global) const {
	return this->Index == nullptr ? global : this->Index[global];
}