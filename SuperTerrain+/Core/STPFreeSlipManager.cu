#pragma once
#include <GPGPU/STPFreeSlipManager.cuh>

using namespace SuperTerrainPlus::STPCompute;

//free-slip data is copied
__host__ STPFreeSlipManager::STPFreeSlipManager(float* texture, const STPFreeSlipData* data) : Data(*data) {
	this->Texture = texture;
}

__host__ STPFreeSlipManager::~STPFreeSlipManager() {

}

__device__ float& STPFreeSlipManager::operator[](unsigned int global) {
	return const_cast<float&>(const_cast<const STPFreeSlipManager*>(this)->operator[](global));
}

__device__ const float& STPFreeSlipManager::operator[](unsigned int global) const {
	return this->Texture[this->operator()(global)];
}

__device__ unsigned int STPFreeSlipManager::operator()(unsigned int global) const {
	return this->Data.GlobalLocalIndex == nullptr ? global : this->Data.GlobalLocalIndex[global];
}