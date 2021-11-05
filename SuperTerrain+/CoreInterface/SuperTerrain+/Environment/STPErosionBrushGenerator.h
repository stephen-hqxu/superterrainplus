#pragma once
#ifndef STP_IMPLEMENTATION
#error __FILE__ is a generator for hydraulic erosion brush and shall not be used externally
#endif

#ifndef STP_RAIN_DROP_SETTING_IMPL
#error __FILE__ is a private PIMPL and can only be included in STPRainDropSetting source file
#endif

//include guard is not needed since this header will only be included by STPRainDropSetting.cpp only once

//Base RainDrop Setting
#include "STPRainDropSetting.cuh"
//Smart Memory
#include <SuperTerrain+/Utility/Memory/STPSmartDeviceMemory.h>

//System ADT
#include <vector>

using namespace SuperTerrainPlus::STPEnvironment;

using glm::uvec2;

/**
 * @brief STPErosionBrushGenerator is a manager for STPRainDropSetting. It generates erosion brush table, manages the memory
 * and export the brush to the rain drop setting for device to use.
*/
class STPRainDropSetting::STPErosionBrushGenerator {
private:

	//so when we copy data to device, we don't need to waste device memory for std::vector
	//Do this after STPHeightfieldSettings no longer inherits this struct (soon after introducing biome)
	//Precomputed erosion brush indices, this is the compute buffer stored on host
	std::vector<int> ErosionBrushIndicesCache;
	//Precomputed erosion brush weights, this is the compute buffer stored on host
	std::vector<float> ErosionBrushWeightsCache;

	//Managed device memory, we will assign the underlying pointer to the base class so it can be used by device directly
	STPSmartDeviceMemory::STPDeviceMemory<int[]> ErosionBrushIndicesDevice;
	STPSmartDeviceMemory::STPDeviceMemory<float[]> ErosionBrushWeightsDevice;

	//The storage for raindrop settings
	STPRainDropSetting& Storage;

	/**
	 * @brief Copy the host erosion brush cache to managed device memory space, and export a pointer to the base class.
	 * The memory exported will be valid so long as:
	 * 1. The instance of STPErosionBrushGenerator the memory belongs to is alive.
	 * 2. Erosion brush is not re-generated
	*/
	void makeDeviceAvailable();

public:

	/**
	 * @brief Init STPErosionBrushGenerator with default STPRainDropSetting settings
	 * @param raindrop_setting The pointer to the raindrop setting storage class, where the result of the generation will be stored
	*/
	STPErosionBrushGenerator(STPRainDropSetting&);

	STPErosionBrushGenerator(const STPErosionBrushGenerator&) = delete;

	STPErosionBrushGenerator(STPErosionBrushGenerator&&) = default;

	STPErosionBrushGenerator& operator=(const STPErosionBrushGenerator&) = delete;

	STPErosionBrushGenerator& operator=(STPErosionBrushGenerator&&) = default;

	~STPErosionBrushGenerator() = default;

	//Init the erosion brush indices and weights
	//@see STPRainDropSetting
	void operator()(uvec2, unsigned int);

};