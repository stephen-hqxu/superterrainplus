#pragma once
#include <Settings/STPRainDropSettings.hpp>

#include <STPDeviceErrorHandler.h>

#include <algorithm>

using namespace SuperTerrainPlus::STPSettings;

__host__ STPRainDropSettings::STPRainDropSettings() : STPSetting() {
	this->ErosionBrushRadius = 0u;
	this->BrushSize = 0u;
	this->RainDropCount = 0u;
	this->Inertia = 0.0f;
	this->SedimentCapacityFactor = 1.0f;
	this->minSedimentCapacity = 0.0f;
	this->initWaterVolume = 1.0f;
	this->minWaterVolume = 0.0f;
	this->Friction = 0.0f;
	this->initSpeed = 0.0f;
	this->ErodeSpeed = 0.0f;
	this->DepositSpeed = 0.0f;
	this->EvaporateSpeed = 0.0f;
	this->Gravity = 1.0f;
	this->ErosionBrushIndices = nullptr;
	this->ErosionBrushWeights = nullptr;
}

__host__ STPRainDropSettings::~STPRainDropSettings() {
	this->omitDeviceAvailable();
}

__host__ void STPRainDropSettings::makeDeviceAvailable() const {
	//Now copy host data to device (store in this struct)
	//check if it has been initialised before, and if so we need to reallocate memory
	this->omitDeviceAvailable();

	//copy the host result to device, since memcpy cannot operate iterator
	int* brushIndex_pinned = nullptr;
	float* brushWeight_pinned = nullptr;
	STPcudaCheckErr(cudaMallocHost(&brushIndex_pinned, sizeof(int) * this->ErosionBrushIndicesCache.size()));
	STPcudaCheckErr(cudaMallocHost(&brushWeight_pinned, sizeof(float) * this->ErosionBrushWeightsCache.size()));
	std::copy(this->ErosionBrushIndicesCache.cbegin(), this->ErosionBrushIndicesCache.cend(), brushIndex_pinned);
	std::copy(this->ErosionBrushWeightsCache.cbegin(), this->ErosionBrushWeightsCache.cend(), brushWeight_pinned);

	STPcudaCheckErr(cudaMalloc(&this->ErosionBrushIndices, sizeof(int) * this->ErosionBrushIndicesCache.size()));
	STPcudaCheckErr(cudaMalloc(&this->ErosionBrushWeights, sizeof(float) * this->ErosionBrushWeightsCache.size()));
	STPcudaCheckErr(cudaMemcpy(this->ErosionBrushIndices, brushIndex_pinned, sizeof(int) * this->ErosionBrushIndicesCache.size(), cudaMemcpyHostToDevice));
	STPcudaCheckErr(cudaMemcpy(this->ErosionBrushWeights, brushWeight_pinned, sizeof(float) * this->ErosionBrushWeightsCache.size(), cudaMemcpyHostToDevice));
	//free the cache
	STPcudaCheckErr(cudaFreeHost(brushIndex_pinned));
	STPcudaCheckErr(cudaFreeHost(brushWeight_pinned));
}

__host__ void STPRainDropSettings::omitDeviceAvailable() const {
	if (this->ErosionBrushIndices != nullptr) {
		STPcudaCheckErr(cudaFree(this->ErosionBrushIndices));
		this->ErosionBrushIndices = nullptr;
	}
	if (this->ErosionBrushWeights != nullptr) {
		STPcudaCheckErr(cudaFree(this->ErosionBrushWeights));
		this->ErosionBrushWeights = nullptr;
	}
}

__host__ bool STPRainDropSettings::validate() const {
	static auto checkRange = []__host__(float value, float lower, float upper) -> bool {
		return value >= lower && value <= upper;
	};

	return checkRange(this->Inertia, 0.0f, 1.0f)
		&& this->SedimentCapacityFactor > 0.0f
		&& this->minSedimentCapacity >= 0.0f
		&& this->initWaterVolume > 0.0f
		&& this->minWaterVolume >= 0.0f
		&& checkRange(this->Friction, 0.0f, 1.0f)
		&& this->initSpeed >= 0.0f
		&& checkRange(this->ErodeSpeed, 0.0f, 1.0f)
		&& checkRange(this->DepositSpeed, 0.0f, 1.0f)
		&& checkRange(this->EvaporateSpeed, 0.0f, 1.0f)
		&& this->Gravity > 0.0f;
}

__host__ void STPRainDropSettings::setErosionBrushRadius(uint2 slipRange, unsigned int erodeRadius) {
	const int radius = static_cast<int>(erodeRadius);
	//radius must be greater than 0
	double weightSum = 0.0f;

	double sqrDst = 0.0f;
	double currentbrushWeight;
	//calculate the brushing weight
	//unfortunately we can't parallel compute this on gpu or multithread cpu since the number of thread to dispatch is undefined
	for (int brushY = -radius; brushY <= radius; brushY++) {
		for (int brushX = -radius; brushX <= radius; brushX++) {
			sqrDst = 1.0 * brushX * brushX + brushY * brushY * 1.0;
			if (sqrDst < radius * radius) {//The brush lies within the erosion range
				this->ErosionBrushIndicesCache.push_back(brushY * slipRange.x + brushX);
				currentbrushWeight = 1 - sqrt(sqrDst) / radius;
				weightSum += currentbrushWeight;
				this->ErosionBrushWeightsCache.push_back(static_cast<float>(currentbrushWeight));
			}
		}
	}
	//normalise the brush weight
	std::for_each(this->ErosionBrushWeightsCache.begin(), this->ErosionBrushWeightsCache.end(), [&weightSum]__host__(float& w) -> void { w /= static_cast<float>(weightSum); });

	//store the brush radius for later computation
	this->ErosionBrushRadius = erodeRadius;
	this->BrushSize = static_cast<unsigned int>(this->ErosionBrushIndicesCache.size());
}

__host__ __device__ unsigned int STPRainDropSettings::getErosionBrushRadius() const {
	return this->ErosionBrushRadius;
}

__host__ __device__ unsigned int STPRainDropSettings::getErosionBrushSize() const {
	return this->BrushSize;
}