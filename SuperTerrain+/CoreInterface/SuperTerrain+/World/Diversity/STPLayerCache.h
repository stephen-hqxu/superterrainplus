#pragma once
#ifndef _STP_LAYER_CACHE_H_
#define _STP_LAYER_CACHE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Biome define
#include "STPBiomeDefine.h"
//System
#include <memory>
#include <optional>
#include <tuple>

namespace SuperTerrainPlus::STPDiversity {
	/**
	 * @brief STPLayerCache is a smart caching system that stores computed layer sample and read directly from when available
	*/
	class STP_API STPLayerCache {
	public:

		//The data to locate an entry in the cache.
		//It contains:
		//True if the address given has data cached under this location, false otherwise.
		//And the rests represents key and index to the entry.
		//If the first flag indicates a false, the key and index can be used to store this data into cache.
		typedef std::tuple<bool, unsigned long long, unsigned long long> STPCacheEntry;
		//Represents a piece of data cached.
		typedef std::optional<Sample> STPCacheData;

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

		~STPLayerCache() = default;

		STPLayerCache& operator=(const STPLayerCache&) = delete;

		STPLayerCache& operator=(STPLayerCache&&) = delete;

		/**
		 * @brief Use the cache entry to read the cached value from the cache.
		 * @param entry A valid cache entry.
		 * @return The cache data associated with the cache entry.
		 * If the entry does not point point to a valid cache to this data, none is returned.
		*/
		STPCacheData read(STPCacheEntry);

		/**
		 * @brief Write a piece of data to the cache entry.
		 * @param entry The cache entry where the data will be written.
		 * Note that data is only written when entry flag is false, indicating the address has no data cached.
		 * @param sample The sample data to be cached.
		*/
		void write(STPCacheEntry, Sample);

		/**
		 * @brief Attempt to find the cache entry using an address specified in the cache.
		 * @param x The x world coordinate.
		 * @param y The y world coordinate.
		 * @param z The z world coordinate.
		 * @return The cache entry to the data.
		*/
		STPCacheEntry locate(int, int, int) const;

		/**
		 * @brief Empty the content of the cache, size is not changed. This operation is not atomic.
		*/
		void clearCache();

		/**
		 * @brief Retrieve the size of the cache
		 * @return The size of the cache
		*/
		unsigned long long getCapacity() const;
	};
}
#endif//_STP_LAYER_CACHE_H_