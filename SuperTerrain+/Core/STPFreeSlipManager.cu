#pragma once
#include <GPGPU/STPFreeSlipManager.cuh>

using namespace SuperTerrainPlus::STPCompute;

//free-slip data is copied
__host__ STPFreeSlipManager::STPFreeSlipManager(float* texture, const STPFreeSlipData* data) :
	Data(data), Texture(texture) {

}

__host__ STPFreeSlipManager::~STPFreeSlipManager() {

}

__device__ __host__ float& STPFreeSlipManager::operator[](unsigned int global) {
	return const_cast<float&>(const_cast<const STPFreeSlipManager*>(this)->operator[](global));
}

__device__ __host__ const float& STPFreeSlipManager::operator[](unsigned int global) const {
	return this->Texture[this->operator()(global)];
}

__device__ __host__ unsigned int STPFreeSlipManager::operator()(unsigned int global) const {
	return this->Data->GlobalLocalIndex == nullptr ? global : this->Data->GlobalLocalIndex[global];
}