//Enable including the header
#define STP_RAIN_DROP_SETTING_IMPL
#include <SuperTerrain+/Environment/STPErosionBrushGenerator.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.h>

//GLM
#include <glm/exponential.hpp>

#include <algorithm>

STPRainDropSetting::STPErosionBrushGenerator::STPErosionBrushGenerator(STPRainDropSetting& raindrop_setting) : Storage(raindrop_setting) {

}

void STPRainDropSetting::STPErosionBrushGenerator::makeDeviceAvailable() {
	//allocate device memory
	//because we are using smart pointer, existing memory will be freed automatically
	this->ErosionBrushIndicesDevice = STPSmartDeviceMemory::makeDevice<int[]>(this->ErosionBrushIndicesCache.size());
	this->ErosionBrushWeightsDevice = STPSmartDeviceMemory::makeDevice<float[]>(this->ErosionBrushWeightsCache.size());

	//copy
	STPcudaCheckErr(cudaMemcpy(this->ErosionBrushIndicesDevice.get(), this->ErosionBrushIndicesCache.data(),
		sizeof(int) * this->ErosionBrushIndicesCache.size(), cudaMemcpyHostToDevice));
	STPcudaCheckErr(cudaMemcpy(this->ErosionBrushWeightsDevice.get(), this->ErosionBrushWeightsCache.data(),
		sizeof(float) * this->ErosionBrushWeightsCache.size(), cudaMemcpyHostToDevice));

	//assign the raw pointer to setting
	this->Storage.ErosionBrushIndices = this->ErosionBrushIndicesDevice.get();
	this->Storage.ErosionBrushWeights = this->ErosionBrushWeightsDevice.get();
}

void STPRainDropSetting::STPErosionBrushGenerator::operator()(uvec2 slipRange, unsigned int erodeRadius) {
	if (erodeRadius == 0u) {
		//radius must be greater than 0
		throw STPException::STPBadNumericRange("Erosion brush radius must be a positive integer");
	}
	if (slipRange.x == 0u || slipRange.y == 0u) {
		//none of the component should be zero
		throw STPException::STPBadNumericRange("Both components in free-slip range should be positive");
	}

	const int radius = static_cast<int>(erodeRadius);
	double weightSum = 0.0;

	double currentbrushWeight;
	//calculate the brushing weight
	for (int brushY = -radius; brushY <= radius; brushY++) {
		for (int brushX = -radius; brushX <= radius; brushX++) {
			if (double sqrDst = 1.0 * brushX * brushX + brushY * brushY * 1.0; 
				sqrDst < radius * radius) {
				//The brush lies within the erosion range
				this->ErosionBrushIndicesCache.emplace_back(brushY * static_cast<int>(slipRange.x) + brushX);
				currentbrushWeight = 1 - glm::sqrt(sqrDst) / radius;
				weightSum += currentbrushWeight;
				this->ErosionBrushWeightsCache.emplace_back(static_cast<float>(currentbrushWeight));
			}
		}
	}
	//normalise the brush weight
	std::transform(this->ErosionBrushWeightsCache.begin(), this->ErosionBrushWeightsCache.end(), this->ErosionBrushWeightsCache.begin(), 
		[weightSum](auto weight) { return weight / static_cast<float>(weightSum); });

	//store the brush radius for later computation, in the base class raindrop setting
	this->Storage.ErosionBrushRadius = erodeRadius;
	this->Storage.BrushSize = static_cast<unsigned int>(this->ErosionBrushIndicesCache.size());

	//update the device memory space
	this->makeDeviceAvailable();
}