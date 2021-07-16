#pragma once
#ifndef _STP_BIOME_FACTORY_H_
#define _STP_BIOME_FACTORY_H_

//System
#include <shared_mutex>
//GLM
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
//Biome
#include "STPBiome.h"
#include "STPLayerManager.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPBiome is a series of biome generation algorithm that allows user to define their own implementations
	*/
	namespace STPBiome {

		/**
		 * @brief STPBiomeFactory provides definition for biome layers and biome lookup table
		*/
		class STPBiomeFactory {
		private:

			//Generate biome layer
			typedef STPLayerManager (*STPManufacturer)();

			mutable std::shared_mutex cache_lock;

			STPManufacturer manufacturer = nullptr;

		public:

			//Specify the dimension of the generated biome map, in 3 dimension
			const glm::uvec3 BiomeDimension;

			/**
			 * @brief Init the biome factory
			 * @param dimension The dimension of the biome map.
			 * If the y component of the dimension is one, a 2D biome map will be generated
			*/
			STPBiomeFactory(glm::uvec3);

			/**
			 * @brief Init biome factory with internal cache memory pool that can be used for multi-threading, each thread will be automatically allocaed one cache
			 * @param dimension The dimension of the biome map
			 * If the y component of the dimension is one, a 2D biome map will be generated
			 * @param manufacturer The biome layer chain generator function
			*/
			STPBiomeFactory(glm::uvec3, STPManufacturer);

			/**
			 * @brief Init the biome factory
			 * @param dimension The dimension of the biome map, this will init a 2D biome map generator, with x and z component only
			*/
			STPBiomeFactory(glm::uvec2);

			/**
			 * @brief Init biome factory with internal cache memory pool that can be used for multi-threading, each thread will be automatically allocaed one cache
			 * @param dimension The dimension of the biome map, this will init a 2D biome map generator, with x and z component only
			 * @param manufacturer The biome layer chain generator function
			*/
			STPBiomeFactory(glm::uvec2, STPManufacturer);

			~STPBiomeFactory();

			/**
			 * @brief Generate a biome map using the biome chain implementation
			 * @param chain The biome chain implementation, it should be the returned value from "manufacture" 
			 * @param offset The offset of the biome map, that is equavalent to the world coordinate.
			 * @return The biome id map, it needs to be freed maunally
			*/
			const Sample* generate(STPLayerManager*, glm::ivec3) const;

		};

	}
}
#endif//_STP_BIOME_FACTORY_H_