#pragma once
#ifndef _STP_HEIGHTFIELD_GENERATOR_CUH_
#define _STP_HEIGHTFIELD_GENERATOR_CUH_

//System
#include <mutex>
//CUDA
//CUDA lib are included in the "Engine" section
#include <curand_kernel.h>
//Engine
#include "STPSimplexNoise.cuh"
#include "STPRainDrop.cuh"
#include "../World/Biome/STPBiome.h"
#include "../Helpers/STPMemoryPool.hpp"
//Settings
#include "../Settings/STPHeightfieldSettings.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief GPGPU compute suites for Super Terrain + program, powered by CUDA
	*/
	namespace STPCompute {
		
		/**
		 * @brief Generate the terriain height map, the height map will be then used to
		 * represent the offset in y direction of the terrain. Heightfield is generated by 2D simplex noise.
		*/
		class STPHeightfieldGenerator {
		public:

			//TODO You can change your preferred RNG here!
			//Choosen generator for curand
			typedef curandStatePhilox4_32_10 curandRNG;

			//A function that converts biome id to the index corresponded in biome table
			//By default it's a 1-1 mapping, meaning biome id = index
			typedef size_t(*STPBiomeInterpreter)(STPBiome::Sample);

		private:

			/**
			 * @brief Memory allocation of tempoary computing cache for heightfield generator
			*/
			class STPHeightfieldAllocator {
			public:

				/**
				 * @brief Allocate memory on GPU
				 * @param count The number of byte of float
				 * @return The device pointer
				*/
				__host__ float* allocate(size_t);

				/**
				 * @brief Free up the GPU memory
				 * @param count The number float to free
				 * @param The device pointer to free
				*/
				__host__ void deallocate(size_t, float*);

			};

			//Launch parameter for texture
			dim3 numThreadperBlock_Map, numBlock_Map;
			//Launch parameter for hydraulic erosion
			int numThreadperBlock_Erosion, numBlock_Erosion;

			/**
			 * @brief Simplex noise generator, on device
			*/
			STPSimplexNoise* simplex = nullptr;
			const STPSimplexNoise simplex_h;
			//All parameters for the noise generator, stoed on host, passing value to device
			const STPSettings::STPSimplexNoiseSettings Noise_Settings;

			//curand random number generator for erosion, each generator will be dedicated for one thread, i.e., thread independency
			curandRNG* RNG_Map = nullptr;
			//Determine the number of raindrop to summon, the higher the more accurate but slower
			//Each time this value changes, the rng needs to be re-sampled
			unsigned int NumRaindrop = 0u;

			STPBiome::STPBiome* BiomeDictionary = nullptr;
			//Temp cache on device for heightmap computation
			mutable std::mutex memorypool_lock;
			mutable STPMemoryPool<float, STPHeightfieldAllocator> MapCache_device;

		public:

			/**
			 * @brief Init the heightfield generator
			 * @param noise_settings Stored all parameters for the heightmap random number generator, it will be deep copied to the class so dynamic memory is not required
			*/
			__host__ STPHeightfieldGenerator(STPSettings::STPSimplexNoiseSettings* const);

			__host__ ~STPHeightfieldGenerator();

			__host__ STPHeightfieldGenerator(const STPHeightfieldGenerator&) = delete;

			__host__ STPHeightfieldGenerator(STPHeightfieldGenerator&&) = delete;

			__host__ STPHeightfieldGenerator& operator=(const STPHeightfieldGenerator&) = delete;

			__host__ STPHeightfieldGenerator& operator=(STPHeightfieldGenerator&&) = delete;

			/**
			 * @brief Load the settings for heightfield generator, all subsequent computation will base on this settings. Settings are copied.
			 * It needs to be called before launching any compute
			 * @param settings The parameters of the generation algorithm. It should be on host side.
			 * Providing no arguement or nullptr will clear all exisiting settings, making it undefined.
			 * @return True if setting can be used
			*/
			__host__ static bool useSettings(const STPSettings::STPHeightfieldSettings* const = nullptr);

			/**
			 * @brief Define the biome dictionary for looking up biome settins according to the biome id. Each entry of biome will be copied to device
			 * @tparam Ite The iterator to the original container with all biomes
			 * @param begin The beginning of the container with biomes
			 * @param end The end of the container with biomes
			 * @return True if copy was successful
			*/
			template<typename Ite>
			__host__ bool defineDictionary(Ite, Ite);

			/**
			 * @brief Generate the terrain heightfield maps, each heightfield contains four maps, being heightmap and normalmap.
			 * All storage spaces must be preallocated with width * height * sizeof(float), with the exception of normalmap, which requires width * height * sizeof(float) * 4.
			 * The function will first generate our epic height map using simplex noise function, using the parameter provided during class init.
			 * The generated heightmap will be in range [0,1]
			 * Then performing hydraulic erosion algorithm to erode the rough terrain into a more natrual form.
			 * The number of iteration must be set via setErosionIterationCUDA() so pre-computation can be done before launching the program.
			 * Lastly it will generate the normal map for the height map, the normalmap is normalised within the range [0,1].
			 * All four maps are kept in floating point pixel format.
			 * @param heightmap A float array that will be used to stored heightmap pixles, must be pre-allocated with at least width * height * sizeof(float), i.e., R32F format
			 * @param normalmap A float array that will be used to stored normnalmap pixles, will be used to store the output of the normal map, must be
			 * pre allocated with at least width * height * 4 byte per channel * 4, i.e., RGBA32F format.
			 * @param offset The x vector specify the offset on x direction of the map and and z on y direction of the map, and the y vector specify the offset on the final result.
			 * The offset parameter will only be applied on the heightmap generation.
			 * @return True if all operation are successful without any errors
			*/
			__host__ bool generateHeightfieldCUDA(float*, float*, float3 = make_float3(0.0f, 0.0f, 0.0f)) const;

			/**
			 * @brief Set the number of raindrop to spawn for each hydraulic erosion run, each time the function is called some recalculation needs to be re-done.
			 * Determine the number of raindrop to summon, the higher the more accurate but slower
			 * @param raindrop_count The number of raindrop (number ofc iteration) for the erosion algorithm
			 * @return True if successsfully updated the count and no error was generated during calculation
			*/
			__host__ bool setErosionIterationCUDA(unsigned int);

			/**
			 * @brief Get the number of iteration for hydraulic erosion
			 * @return The number of raindrop to erode the terrain
			*/
			__host__ unsigned int getErosionIteration() const;

		};

	}
}
#include "STPHeightfieldGenerator.inl"
#endif//_STP_HEIGHTFIELD_GENERATOR_CUH_