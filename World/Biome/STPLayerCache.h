#pragma once
#ifndef _STP_LAYER_CACHE_H_
#define _STP_LAYER_CACHE_H_

//Functional
#include <functional>
//Exception
#include <stdexcept>
//Biome define
#include "STPBiomeDefine.h"
//System
#include <memory>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPDiversity is a series of biome generation algorithm that allows user to define their own implementations
	*/
	namespace STPDiversity {
		/**
		 * @brief STPLayerCache is a smart caching system that stores computed layer sample and read directly from when available
		*/
		class STPLayerCache final {
		private:

			//Store the key value for a coordinate
			std::unique_ptr<unsigned long long[]> Key;
			//Store the sample number for a layer for a coordinate
			std::unique_ptr<Sample[]> Value;
			//Mask is used to round the key value such that it will be suitable for looking up value in the hash table
			unsigned long long Mask;

			/**
			 * @brief Check if the number is power of 2
			 * @param val The value to check
			 * @return True if the number is power of 2
			*/
			static bool isPow2(unsigned long long);

			/**
			 * @brief Get the mask value by evaluating power of 2 and minus one
			 * @param bits The power to raise
			 * @return The mask value
			*/
			static unsigned long long getMask(unsigned long long);

			/**
			 * @brief Hash the coordinate and generate a unique hash value
			 * @param x The x world coordinate
			 * @param y The y world coordinate
			 * @param z The z world coordinate
			 * @return The unique hash value for this coordinate
			*/
			static unsigned long long uniqueHash(int, int, int);

			/**
			 * @brief An algorithm to convert key value to a raw index value in order to locate the sample value in the hash table
			 * @param key The value of the key
			 * @return The raw index value, it's not the same as index, do remember to limit the range of the index using mask
			*/
			static unsigned long long mixKey(unsigned long long);

		public:

			/**
			 * @brief Init STPLayerCache with allocated spaces
			 * @param capacity The capacity of the cache, it must be power of 2
			*/
			STPLayerCache(size_t);

			STPLayerCache(const STPLayerCache&) = delete;

			STPLayerCache(STPLayerCache&&) = delete;

			~STPLayerCache();

			STPLayerCache& operator=(const STPLayerCache&) = delete;

			STPLayerCache& operator=(STPLayerCache&&) = delete;

			/**
			 * @brief Find the value with coordinate specified in the cache. If found, read directly; if not, it will be evaluated using the sample function
			 * then store into the cache.
			 * This method is NOT thread safe
			 * @param x The x world coordinate
			 * @param y The y world coordinate
			 * @param z The z world coordinate
			 * @param sampler If the cache is not found, it will be evaluated using this function
			 * @return The sample associated with the coordinate
			*/
			Sample cache(int, int, int, std::function<Sample(int, int, int)>);

			/**
			 * @brief Empty the content of the cache, size is not changed. This operation is not atomic.
			*/
			void clearCache();

			/**
			 * @brief Retrieve the size of the cache
			 * @return The size of the cache
			*/
			size_t getCapacity() const;
		};
	}
}
#endif//_STP_LAYER_CACHE_H_