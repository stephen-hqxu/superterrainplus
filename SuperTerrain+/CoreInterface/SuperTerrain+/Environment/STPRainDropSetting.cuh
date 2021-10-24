#pragma once
#ifndef _STP_RAIN_DROP_SETTING_CUH_
#define _STP_RAIN_DROP_SETTING_CUH_

#include <SuperTerrain+/STPCoreDefine.h>
#include "STPSetting.hpp"
//CUDA Runtime
#include <cuda_runtime.h>
//System
#include <memory>
//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief STPRainDropSetting stores all calculation parameters for the rain drops.
	 * Copy and move operation on this class is unsafe if device memory has been made available, and only shallow copy is performed.
	 * To make device memory available across all copies, only call makeAvailable() function on the object that has been copied.
	*/
	struct STP_API STPRainDropSetting : public STPSetting {
	private:

		//Implementation for erosion brush generation
		class STPErosionBrushGenerator;
		std::unique_ptr<STPErosionBrushGenerator> ErosionBrushGenerator;

		//Determine the radius of the droplet that can brush out sediment
		//Specify the radius of the brush. Determines the radius in which sediment is taken from therock layer.
		//The smaller radius is, the deeperand more distinct the ravines will be.
		unsigned int ErosionBrushRadius;
		//Precompted the number of brush
		unsigned int BrushSize;
		//Precomputed erosion brush indices, must be made available so it can be used on device.
		int* ErosionBrushIndices;
		//Precomputed erosion brush weights, must be made available so it can be used on device.
		float* ErosionBrushWeights;

	public:

		//The number of raindrop presented to perform hydraulic erosion
		unsigned int RainDropCount;
		//At zero, water will instanly change direction to flow downhill. At one, water will never change direction. Ranged [0,1]
		float Inertia;
		//Multiplier for how much sediment a droplet can carry.
		float SedimentCapacityFactor;
		//Used to prevent carry capacity getting too close to zero on flatten the entire terrain.
		float minSedimentCapacity;
		//The starting volume of the water droplet
		float initWaterVolume;
		//Used to check when to end the droplet's life time. If the current volme falls below 
		float minWaterVolume;
		//The greater the friction, the quicker the droplet will slow down. Ranged [0,1]
		float Friction;
		//The starting speed of the water droplet
		float initSpeed;
		//How fast a droplet will be filled with sediment. Ranged [0,1]
		float ErodeSpeed;
		//Limit the speed of sediment drop if exceeds the minSedimentCapacity. Ranged [0,1]
		float DepositSpeed;
		//Control how fast the water will evaportate in the droplet. Ranged [0,1]
		float EvaporateSpeed;
		//Control how fast water droplet descends.
		float Gravity;

		/**
		 * @brief Init STPRainDropSettings with default values.
		*/
		__host__ STPRainDropSetting();

		__host__ STPRainDropSetting(const STPRainDropSetting&) = delete;

		__host__ STPRainDropSetting(STPRainDropSetting&&) noexcept;

		__host__ STPRainDropSetting& operator=(const STPRainDropSetting&) = delete;

		__host__ STPRainDropSetting& operator=(STPRainDropSetting&&) noexcept;

		__host__ virtual ~STPRainDropSetting();

		__host__ virtual bool validate() const override;

		/**
		 * @brief Init the erosion brush indices and weights, so each droplet can erode a certain range of terrain but not only the current pixel.
		 * The erosion radius will be automatically updated to the parameter.
		 * @param slipRange The area in both direction where raindrop can slip freely. Usually this is the same as the size of the heightmap, or
		 * under free-slip hydraulic erosion, this is the free-slip range.
		 * No reference is retained after the function returns.
		 * @param erodeRadius Specify the radius of the brush. Determines the radius in which sediment is taken from therock layer.
		 * The smaller radius is, the deeper and more distinct the ravines will be.
		 * Raising the erosion radius also increases the computational time needed for each drop drastically.
		*/
		__host__ void setErosionBrushRadius(glm::uvec2, unsigned int);

		/**
		 * @brief Get the radius of the erosion brush
		 * @return The radius of the erosion brush
		*/
		__host__ __device__ unsigned int getErosionBrushRadius() const;

		/**
		 * @brief Get the number of erosion brush
		 * @return The number of erosion brush
		*/
		__host__ __device__ unsigned int getErosionBrushSize() const;

		/**
		 * @brief Get the device pointer to the erosion brush indices
		 * @return The pointer to the erosion brush indices on device
		*/
		__device__ int* getErosionBrushIndices() const;

		/**
		 * @brief Get the device pointer to the erosion brush weights
		 * @return The pointer to the erosion brush weights on device
		*/
		__device__ float* getErosionBrushWeights() const;

	};

}
#endif//_STP_RAIN_DROP_SETTING_CUH_