#include <SuperTerrain+/World/Chunk/STPErosionBrushGenerator.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Utility/STPDeviceErrorHandler.hpp>

//Container
#include <vector>

//GLM
#include <glm/exponential.hpp>

#include <algorithm>

using std::vector;

using namespace SuperTerrainPlus;

STPErosionBrushGenerator::STPErosionBrushGenerator(unsigned int freeslip_rangeX, unsigned int erosion_radius) {
	if (erosion_radius == 0u) {
		//radius must be greater than 0
		throw STPException::STPBadNumericRange("Erosion brush radius must be a positive integer");
	}
	if (freeslip_rangeX == 0u) {
		//none of the component should be zero
		throw STPException::STPBadNumericRange("Free-slip range row count should be positive");
	}

	/* -------------------------------------- Generate Erosion Brush ------------------------------- */
	const int radius = static_cast<int>(erosion_radius),
		radiusSqr = radius * radius;
	double weightSum = 0.0;
	//temporary cache for generation
	vector<int> indexCache;
	vector<float> weightCache;


	//calculate the brushing weight
	for (int brushY = -radius; brushY <= radius; brushY++) {
		for (int brushX = -radius; brushX <= radius; brushX++) {
			if (double sqrDst = 1.0 * brushX * brushX + brushY * brushY * 1.0;
				sqrDst < radiusSqr) {
				//The brush lies within the erosion range
				const double currentbrushWeight = 1.0 - glm::sqrt(sqrDst) / radius;
				weightSum += currentbrushWeight;
				//store
				indexCache.emplace_back(brushY * static_cast<int>(freeslip_rangeX) + brushX);
				weightCache.emplace_back(static_cast<float>(currentbrushWeight));
			}
		}
	}
	//normalise the brush weight
	std::transform(weightCache.begin(), weightCache.end(), weightCache.begin(),
		[weightSum](auto weight) { return weight / static_cast<float>(weightSum); });

	/* ------------------------------------ Populate Device Memory ---------------------------------------------- */
	//because we are using smart pointer, existing memory will be freed automatically
	this->ErosionBrushIndex_d = STPSmartDeviceMemory::makeDevice<int[]>(indexCache.size());
	this->ErosionBrushWeight_d = STPSmartDeviceMemory::makeDevice<float[]>(weightCache.size());

	//copy
	STP_CHECK_CUDA(cudaMemcpy(this->ErosionBrushIndex_d.get(), indexCache.data(),
		sizeof(int) * indexCache.size(), cudaMemcpyHostToDevice));
	STP_CHECK_CUDA(cudaMemcpy(this->ErosionBrushWeight_d.get(), weightCache.data(),
		sizeof(float) * weightCache.size(), cudaMemcpyHostToDevice));

	//store data
	this->Brush = STPErosionBrush {
		this->ErosionBrushIndex_d.get(),
		this->ErosionBrushWeight_d.get(),
		//index and brush have the same size
		static_cast<unsigned int>(indexCache.size())
	};
}

const STPErosionBrush& STPErosionBrushGenerator::getBrush() const {
	return this->Brush;
}