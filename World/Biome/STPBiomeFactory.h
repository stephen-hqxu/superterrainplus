#pragma once
#ifndef _STP_BIOME_FACTORY_H_
#define _STP_BIOME_FACTORY_H_

//Hand made memory pool
#include "../../Helpers/STPMemoryPool.hpp"
//System
#include <shared_mutex>
//GLM
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
//Biome
#include "STPBiome.h"
#include "STPLayer.h"

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
			typedef STPLayer* (*STPManufacturer)();

			/**
			 * @brief STPBiomeAllocator is the memory allcator for layers
			*/
			class STPBiomeAllocator {
			public:

				/**
				 * @brief Allocate memory for biome layer chain and construct in place
				 * @prarm size Amount of memory
				 * @param manufacturer The function to construct memory
				 * @return The new layer memory
				*/
				STPLayer* allocate(size_t, STPManufacturer);

				/**
				 * @brief Free up the biome layer memory
				 * @param size Amount of memory to free
				 * @param layer The pointer to the biome layer that needs to be freed
				*/
				void deallocate(size_t size, STPLayer*);

			};

			//Cache pool for layers, such that it can be used in multi-threading
			STPMemoryPool<STPLayer, STPBiomeAllocator> layer_cache;
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
			 * @brief Get the number of cache in the internal cache pool
			 * @return The number of cache. If no cache is initialsed or it has been depleted by execution, return 0
			*/
			size_t size() const;

			/**
			 * @brief Generate a biome map using the biome chain implementation
			 * @param chain The biome chain implementation, it should be the returned value from "manufacture" 
			 * @param offset The offset of the biome map, that is equavalent to the world coordinate.
			 * @return The biome id map, it needs to be freed maunally
			*/
			const Sample* generate(STPLayer* const, glm::ivec3) const;
			
			/**
			 * @brief Generate a biome map using the internal stored biome chain with cache.
			 * To avoid spin-locking, it's recommended to call this method from a non-executing thread.
			 * @param offset The offset of the biome map, that is equavalent to the world coordinate.
			 * @return The biome id map, it needs to be freed maunally
			*/
			const Sample* generate(glm::ivec3);

			/**
			 * @brief Free up the storage of a biome map generated
			 * @param map The biome map to free
			*/
			static void dump(Sample*);

		};

	}
}
#endif//_STP_BIOME_FACTORY_H_