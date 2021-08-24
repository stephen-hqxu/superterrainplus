#pragma once
#include <Environment/STPRainDropSetting.h>

#include <SuperError+/STPDeviceErrorHandler.h>

#include <algorithm>

using namespace SuperTerrainPlus::STPEnvironment;

__host__ STPRainDropSetting::STPRainDropSetting() : 
	STPSetting(), 
	ErosionBrushRadius(0u), 
	BrushSize(0u), 
	RainDropCount(0u),
	Inertia(0.0f), 
	SedimentCapacityFactor(1.0f), 
	minSedimentCapacity(0.0f), 
	initWaterVolume(1.0f), 
	minWaterVolume(0.0f), 
	Friction(0.0f), 
	initSpeed(0.0f), 
	ErodeSpeed(0.0f), 
	DepositSpeed(0.0f), 
	EvaporateSpeed(0.0f), 
	Gravity(1.0f), 
	ErosionBrushIndices(nullptr), 
	ErosionBrushWeights(nullptr) {

}

__host__ STPRainDropSetting::~STPRainDropSetting() {
	this->omitDeviceAvailable();
}

__host__ void STPRainDropSetting::makeDeviceAvailable() const {
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

__host__ void STPRainDropSetting::omitDeviceAvailable() const {
	if (this->ErosionBrushIndices != nullptr) {
		STPcudaCheckErr(cudaFree(this->ErosionBrushIndices));
		this->ErosionBrushIndices = nullptr;
	}
	if (this->ErosionBrushWeights != nullptr) {
		STPcudaCheckErr(cudaFree(this->ErosionBrushWeights));
		this->ErosionBrushWeights = nullptr;
	}
}

__host__ bool STPRainDropSetting::validate() const {
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

__host__ void STPRainDropSetting::setErosionBrushRadius(uint2 slipRange, unsigned int erodeRadius) {
	const int radius = static_cast<int>(erodeRadius);
	//radius must be greater than 0
	double weightSum = 0.0f;

	double currentbrushWeight;
	//calculate the brushing weight
	//unfortunately we can't parallel compute this on gpu or multithread cpu since the number of thread to dispatch is undefined
	for (int brushY = -radius; brushY <= radius; brushY++) {
		for (int brushX = -radius; brushX <= radius; brushX++) {
			if (double sqrDst = 1.0 * brushX * brushX + brushY * brushY * 1.0; 
				sqrDst < radius * radius) {//The brush lies within the erosion range
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

__host__ __device__ unsigned int STPRainDropSetting::getErosionBrushRadius() const {
	return this->ErosionBrushRadius;
}

__host__ __device__ unsigned int STPRainDropSetting::getErosionBrushSize() const {
	return this->BrushSize;
}