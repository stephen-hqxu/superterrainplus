#pragma once
#ifndef _STP_HEIGHTFIELD_GENERATOR_H_
#define _STP_HEIGHTFIELD_GENERATOR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <mutex>
#include <vector>
#include <queue>
//CUDA
//CUDA lib are included in the "Engine" section
#include <curand_kernel.h>
//Engine
#include "STPDiversityGenerator.hpp"
#include "./FreeSlip/STPFreeSlipGenerator.h"
#include "../../Utility/Memory/STPSmartStream.h"
#include "../../Utility/Memory/STPSmartDeviceMemory.h"
//Settings
#include "../../Environment/STPHeightfieldSetting.h"
#include "../../Environment/STPChunkSetting.h"

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPCompute {

	/**
	 * @brief Generate the terriain height map, the height map will be then used to
	 * represent the offset in y direction of the terrain. Heightfield is generated by diversity generator.
	*/
	class STP_API STPHeightfieldGenerator {
	public:

		//STPGeneratorOperation controls the operations to perform during heightfield generation
		typedef unsigned short STPGeneratorOperation;

		//TODO You can change your preferred RNG here!
		//Choosen generator for curand
		typedef curandStatePhilox4_32_10 curandRNG;

		//Generate a new heightmap and store the result in the provided memory space
		constexpr static STPGeneratorOperation HeightmapGeneration = 1u << 0u;
		//Erode the heightmap. If HeightmapGeneration flag is not enabled, an available heightmap needs to be provided for the operation
		constexpr static STPGeneratorOperation Erosion = 1u << 1u;
		//Generate normal map and integrate into heightfield. If HeightmapGeneration flag is not enabled, an available heightmap needs to be provided for the operation
		//RGB channel will then contain normalmap and A channel contains heightmap
		//Then format the heightfield map from FP32 to INT16.
		constexpr static STPGeneratorOperation RenderingBufferGeneration = 1u << 2u;

		/**
		 * @brief STPMapStorage stores heightfield data for the generator
		*/
		struct STPMapStorage {
		public:

			//- A Sample array (sample is implementation defined, usually it's uint16) where biomemap is located.
			//- the number of biomemap should be the same as that in Heightmap32F when erosion is turned on.
			//We need free-slip biomemap so custom heightmap implementation can do biome-edge interpolation
			//See documentation of Heightmap32F for more details
			//If heightmap generation is not enabled, no biomemap is required
			std::vector<STPDiversity::Sample*> Biomemap;
			//- A float array that will be used to stored heightmap pixles, must be pre-allocated with at least width * height * sizeof(float), i.e., R32F format
			//- If generator is instructed to generate only a single heightmap, only one map is required
			//- If hydraulic erosion and/or normalmap generation is enabled, a list of maps of neighbour chunks are required for edge sync, heightmap generation will 
			//only affect the central chunk, for neighbour chunks it must be precomputed with heightmap to be able to perform free-slip hydraulic erosion,
			//If free-slip hydraulic erosion is disabled, no neighbour chunks are required.
			//- The map pointers should be arranged in row major matrix, with defined neighbour dimension.
			std::vector<float*> Heightmap32F;
			//The x vector specify the offset on x direction of the map and and z on y direction of the map.
			//The offset parameter will only be applied on the heightmap generation.
			glm::vec2 HeightmapOffset = glm::vec2(0.0f);
			//A INT16 array that will be used to stored the heightmap and normalmap after formation. The final format will become R16.
			//The number of pointer provided should be the same as the number of heightmap and normalmap.
			std::vector<unsigned short*> Heightfield16UI;

		};

	private:

		//multi-biome heightmap generator linked with external
		const STPDiversityGenerator& generateHeightmap;
		//heightfield generation parameters
		const STPEnvironment::STPHeightfieldSetting& Heightfield_Setting_h;
		STPSmartDeviceMemory::STPDeviceMemory<STPEnvironment::STPHeightfieldSetting> Heightfield_Setting_d;

		//curand random number generator for erosion, each generator will be dedicated for one thread, i.e., thread independency
		STPSmartDeviceMemory::STPDeviceMemory<curandRNG[]> RNG_Map;
		//free-slip index table generator
		STPFreeSlipGenerator FreeSlipTable;
		STPFreeSlipTextureAttribute TextureBufferAttr;

		//Temp cache on device for heightmap computation
		mutable std::mutex StreamPool_lock;
		mutable cudaMemPool_t MapCacheDevice;
		mutable std::queue<STPSmartStream> StreamPool;

	public:

		/**
		 * @brief Init the heightfield generator
		 * @param chunk_settings All parameters for the chunk to be linked with this generator
		 * @param heightfield_settings All parameters for heightfield generation to be linked with this generator
		 * @param diversity_generator A generator responsible for generating a multi-biome heightmap
		 * @param hint_level_of_concurrency The average numebr of thread that will be used to issue commands to this class.
		 * It's used to assume the size of memory pool to allocate.
		*/
		STPHeightfieldGenerator(const STPEnvironment::STPChunkSetting&, const STPEnvironment::STPHeightfieldSetting&,
			const STPDiversityGenerator&, unsigned int);

		~STPHeightfieldGenerator();

		STPHeightfieldGenerator(const STPHeightfieldGenerator&) = delete;

		STPHeightfieldGenerator(STPHeightfieldGenerator&&) = delete;

		STPHeightfieldGenerator& operator=(const STPHeightfieldGenerator&) = delete;

		STPHeightfieldGenerator& operator=(STPHeightfieldGenerator&&) = delete;

		/**
		 * @brief Generate the terrain heightfield maps, each heightfield contains four maps, being heightmap and normalmap.
		 * All storage spaces must be preallocated with width * height * sizeof(float), with the exception of normalmap, which requires width * height * sizeof(float) * 4.
		 * The function will first generate our epic height map diversity generator, using the parameter provided during class init.
		 * The generated heightmap will be in range [0,1]
		 * Then performing hydraulic erosion algorithm to erode the rough terrain into a more natrual form.
		 * The number of iteration must be set via setErosionIterationCUDA() so pre-computation can be done before launching the program.
		 * Lastly it will generate the normal map for the height map, the normalmap is normalised within the range [0,1].
		 * All four maps are kept in floating point pixel format.
		 * @param args The generator data, see STPMapStorage documentation for more details
		 * @param operation Control what type of operation generator does
		 * @return True if all operations are successful without any errors
		*/
		void operator()(STPMapStorage&, STPGeneratorOperation) const;

	};

}
#endif//_STP_HEIGHTFIELD_GENERATOR_H_