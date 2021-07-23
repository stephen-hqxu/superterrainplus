#pragma once
#ifndef _STP_DIVERSITY_GENERATOR_HPP_
#define _STP_DIVERSITY_GENERATOR_HPP_

//CUDA Runtime
#include <cuda_runtime.h>
//Biome Defines
#include "../World/Biome/STPBiomeDefine.h"

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
		 * @brief STPDiversityGenerator is a base class which provides a programmable interface for customised multi-biome heightmap generation
		*/
		class STPDiversityGenerator {
		protected:

			/**
			 * @brief Init diversity generator
			*/
			STPDiversityGenerator() = default;

			virtual ~STPDiversityGenerator() = default;

		public:

			/**
			 * @brief Generate a biome-specific heightmaps
			 * @param heightmap The result of generated heightmap that will be stored
			 * @param biomemap The biomemap, which is an array of biomeID, the meaning of biomeID is however implementation-specific
			 * @param offset The offset of maps in world coordinate
			 * @param stream The stream currently being used
			*/
			virtual void operator()(float*, const STPDiversity::Sample*, float2, cudaStream_t) const = 0;

		};

	}
}
#endif//_STP_DIVERSITY_GENERATOR_HPP_