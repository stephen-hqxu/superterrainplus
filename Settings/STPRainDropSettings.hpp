#pragma once
#ifndef _STP_RAIN_DROP_SETTINGS_HPP_
#define _STP_RAIN_DROP_SETTINGS_HPP_

#include "STPSettings.hpp"
//CUDA function flags
#include <cuda_runtime_api.h>
//CUDA vector
#include <vector_functions.h>
//Runtime
#include <cuda_runtime.h>
//System ADT
#include <list>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPSettings contains all configurations for each generators, like heightmap, normalmap, biomes, texture, etc.
	*/
	namespace STPSettings {

		/**
		 * @brief STPRainDropPara stores all calculation parameters for the rain drops.
		*/
		struct STPRainDropSettings: public STPSettings {
		private:

			//Determine the radius of the droplet that can brush out sediment
			//Specify the radius of the brush. Determines the radius in which sediment is taken from therock layer.
			//The smaller radius is, the deeperand more distinct the ravines will be.
			//InitErosionBrush() must be called to set the value in order to compute the erosion brush, sololy setting the radius will not take effect
			unsigned int ErosionBrushRadius;
			//Precompted the number of brush
			unsigned int BrushSize;

		public:

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
			//Precomputed erosion brush indices, stored on device.
			int* ErosionBrushIndices;
			//Precomputed erosion brush weights, stored on device.
			float* ErosionBrushWeights;

		private:

			/**
			 * @brief Deep copy the pointer only
			 * @param src The source from where the copy will happen
			*/
			void performCopyptr(const STPRainDropSettings& src) {
				//performing deep copy for the device memory
				if (src.ErosionBrushIndices != nullptr) {
					cudaMalloc(&this->ErosionBrushIndices, sizeof(int) * this->BrushSize);
					cudaMemcpy(this->ErosionBrushIndices, src.ErosionBrushIndices, sizeof(int) * this->BrushSize, cudaMemcpyDeviceToDevice);
				}
				else {
					this->ErosionBrushIndices = nullptr;
				}
				if (src.ErosionBrushWeights != nullptr) {
					cudaMalloc(&this->ErosionBrushWeights, sizeof(float) * this->BrushSize);
					cudaMemcpy(this->ErosionBrushWeights, src.ErosionBrushWeights, sizeof(float) * this->BrushSize, cudaMemcpyDeviceToDevice);
				}
				else {
					this->ErosionBrushWeights = nullptr;
				}
			}

		public:

			/**
			 * @brief Init STPRainDropSettings with default values.
			*/
			__host__ STPRainDropSettings() : STPSettings() {
				this->ErosionBrushRadius = 0u;
				this->BrushSize = 0u;
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

			//Performing deep copy for the parameter (device memory of the erosion brush will be copied as well)
			__host__ STPRainDropSettings(const STPRainDropSettings& obj)
				: ErosionBrushRadius(obj.ErosionBrushRadius), BrushSize(obj.BrushSize), Inertia(obj.Inertia)
				, SedimentCapacityFactor(obj.SedimentCapacityFactor), minSedimentCapacity(obj.minSedimentCapacity), initWaterVolume(obj.initWaterVolume)
				, minWaterVolume(obj.minWaterVolume), Friction(obj.Friction), initSpeed(obj.initSpeed), ErodeSpeed(obj.ErodeSpeed)
				, DepositSpeed(obj.DepositSpeed), EvaporateSpeed(obj.EvaporateSpeed), Gravity(obj.Gravity) {
				//performing deep copy for the device memory
				this->performCopyptr(obj);
			}

			__host__ ~STPRainDropSettings() {
				if (this->ErosionBrushIndices != nullptr) {
					cudaFree(this->ErosionBrushIndices);
				}
				if (this->ErosionBrushWeights != nullptr) {
					cudaFree(this->ErosionBrushWeights);
				}
			}

			/**
			 * @brief Perform a deep copy on the targeting source
			 * @param src The source from which to copy
			 * @return Modified class object
			*/
			__host__ STPRainDropSettings& operator=(const STPRainDropSettings& src) {
				this->ErosionBrushRadius = src.ErosionBrushRadius;
				this->BrushSize = src.BrushSize;
				this->Inertia = src.Inertia;
				this->SedimentCapacityFactor = src.SedimentCapacityFactor;
				this->minSedimentCapacity = src.minSedimentCapacity;
				this->initWaterVolume = src.initWaterVolume;
				this->minWaterVolume = src.minWaterVolume;
				this->Friction = src.Friction;
				this->initSpeed = src.initSpeed;
				this->ErodeSpeed = src.ErodeSpeed;
				this->DepositSpeed = src.DepositSpeed;
				this->EvaporateSpeed = src.EvaporateSpeed;
				this->Gravity = src.Gravity;
				//deep copy the device side erosion brush and indices
				this->performCopyptr(src);
				
				return *this;
			}

			__host__ virtual bool validate() override {
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

			/**
			 * @brief Init the erosion brush indices and weights, so each droplet can erode a certain range of terrain but not only the current pixel.
			 * The erosion radius will be automatically updated to the parameter.
			 * @param mapSize The size of the heightmap
			 * @param erodeRadius Specify the radius of the brush. Determines the radius in which sediment is taken from therock layer.
			 * The smaller radius is, the deeper and more distinct the ravines will be.
			 * Raising the erosion radius also increases the computational time needed for each drop drastically.
			*/
			__host__ void setErosionBrushRadius(uint2 mapSize, unsigned int erodeRadius) {
				const int radius = static_cast<int>(erodeRadius);
				//radius must be greater than 0
				//Getting the storage on host
				std::list<int> brushIndex;
				std::list<float> brushWeight;
				float weightSum = 0.0f;

				float sqrDst = 0.0f;
				float currentbrushWeight;
				//calculate the brushing weight
				//unfortunately we can't parallel compute this on gpu or multithread cpu since the number of thread to dispatch is undefined
				for (int brushY = -radius; brushY <= radius; brushY++) {
					for (int brushX = -radius; brushX <= radius; brushX++) {
						sqrDst = 1.0f * brushX * brushX + brushY * brushY * 1.0f;
						if (sqrDst < radius * radius) {//The brush lies within the erosion range
							brushIndex.push_back(brushY * mapSize.x + brushX);
							currentbrushWeight = 1 - sqrt(sqrDst) / radius;
							weightSum += currentbrushWeight;
							brushWeight.push_back(currentbrushWeight);
						}
					}
				}
				//normalise the brush weight
				for (std::list<float>::iterator it = brushWeight.begin(); it != brushWeight.end(); it++) {
					*it /= weightSum;
				}
				this->BrushSize = static_cast<unsigned int>(brushIndex.size());

				//Now copy host data to device (store in this struct)
				//check if it has been initialised before, and if so we need to reallocate memory
				if (this->ErosionBrushIndices != nullptr) {
					cudaFree(this->ErosionBrushIndices);
				}
				if (this->ErosionBrushWeights != nullptr) {
					cudaFree(this->ErosionBrushWeights);
				}

				//copy the host result to device, since memcpy cannot operate iterator (dereferencing iterator and treat it as pointer is dangerous, especially when we are working with linked list)
				int* brushIndex_pinned = nullptr;
				float* brushWeight_pinned = nullptr;
				cudaMallocHost(&brushIndex_pinned, sizeof(int) * brushIndex.size());
				cudaMallocHost(&brushWeight_pinned, sizeof(float) * brushWeight.size());
				std::copy(brushIndex.begin(), brushIndex.end(), brushIndex_pinned);
				std::copy(brushWeight.begin(), brushWeight.end(), brushWeight_pinned);

				cudaMalloc(&this->ErosionBrushIndices, sizeof(int) * brushIndex.size());
				cudaMalloc(&this->ErosionBrushWeights, sizeof(float) * brushWeight.size());
				cudaMemcpy(this->ErosionBrushIndices, brushIndex_pinned, sizeof(int) * brushIndex.size(), cudaMemcpyHostToDevice);
				cudaMemcpy(this->ErosionBrushWeights, brushWeight_pinned, sizeof(float) * brushWeight.size(), cudaMemcpyHostToDevice);
				//free the cache
				cudaFreeHost(brushIndex_pinned);
				cudaFreeHost(brushWeight_pinned);

				//store the brush radius for later computation
				this->ErosionBrushRadius = erodeRadius;
			}

			/**
			 * @brief Get the radius of the erosion brush
			 * @return The radius of the erosion brush
			*/
			__host__ __device__ unsigned int getErosionBrushRadius() {
				return this->ErosionBrushRadius;
			}

			/**
			 * @brief Get the number of erosion brush
			 * @return The number of erosion brush
			*/
			__host__ __device__ unsigned int getErosionBrushSize() {
				return this->BrushSize;
			}
		};

	}

}
#endif//_STP_RAIN_DROP_SETTINGS_HPP_