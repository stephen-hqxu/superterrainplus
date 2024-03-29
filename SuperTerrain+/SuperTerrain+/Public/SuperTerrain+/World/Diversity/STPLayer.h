#pragma once
#ifndef _STP_LAYER_H_
#define _STP_LAYER_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "../STPWorldMapPixelFormat.hpp"

//System
#include <utility>
#include <memory>

#include <initializer_list>
#include <optional>

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief Super Terrain Plus uses layered biome generation algorithm (ref: Minecraft),
	 * STPLayer provides a abstract base class for implementations of subsequent layers.
	*/
	class STP_API STPLayer {
	public:

		/**
		 * @brief STPLocalSampler is a random number generator for using a local seed.
		 * Since local seed is specific to a coordinate in a layer, the local sampler guarantees to return the same sequence for
		 * the same coordinate on the same layer.
		*/
		class STP_API STPLocalSampler {
		private:

			//Layer seed for a particular layer implementation
			const STPSeed_t LayerSeed;
			//local seed varies from world coordinate, but the same coordinate will always yield the same local seed
			mutable STPSeed_t LocalSeed;

		public:

			/**
			 * @brief Init local sampler.
			 * @param layer_seed Layer seed that is unique to each layer.
			 * @param local_seed Local seed that is unique to each world coordinate for every layer.
			*/
			STPLocalSampler(STPSeed_t, STPSeed_t) noexcept;

			~STPLocalSampler() = default;

			/**
			 * @brief Get the next random number in the sequence.
			 * @param range The range from 0u to the specified value.
			 * @return The next sequence.
			*/
			STPSample_t nextValue(STPSample_t) const noexcept;

			/**
			 * @brief Select randomly among two variables.
			 * @param var1 The first variable to choose.
			 * @param var2 The second variable to choose.
			 * @return The chosen value of the variable.
			*/
			STPSample_t choose(STPSample_t, STPSample_t) const noexcept;

			/**
			 * @brief Select randomly among four variables.
			 * @param var1 The first variable to choose.
			 * @param vaw2 The second variable to choose.
			 * @param var3 The third variable to choose.
			 * @param vaw4 The forth variable to choose.
			 * @return The chosen value of the variable.
			*/
			STPSample_t choose(STPSample_t, STPSample_t, STPSample_t, STPSample_t) const noexcept;

		};

		//The number of ascendant
		const size_t AscendantCount;

	private:

		/**
		 * @brief STPLayerCache is a smart caching system that stores computed layer sample and read directly from when available.
		*/
		class STPLayerCache {
		private:

			STPLayer& Layer;

			//Store the key value for a coordinate
			const std::unique_ptr<unsigned long long[]> Key;
			//Store the sample number for a layer for a coordinate
			const std::unique_ptr<STPSample_t[]> Value;
			//Mask is used to round the key value such that it will be suitable for looking up value in the hash table
			const unsigned long long Mask;

		public:

			/**
			 * @brief Init STPLayerCache with allocated spaces.
			 * @param layer The dependent layer.
			 * @param capacity The capacity of the cache, it must be power of 2.
			*/
			STPLayerCache(STPLayer&, size_t);

			STPLayerCache(const STPLayerCache&) = delete;

			STPLayerCache(STPLayerCache&&) = delete;

			STPLayerCache& operator=(const STPLayerCache&) = delete;

			STPLayerCache& operator=(STPLayerCache&&) = delete;

			~STPLayerCache() = default;

			/**
			 * @brief Locate the cache entry associated with the world coordinate given.
			 * If cache is found, read directly from the cache;
			 * otherwise it will invoke the layer sample function to compute the value, and store into cache entry.
			 * @param x The X world coordinate.
			 * @param y The Y world coordinate.
			 * @param z THe Z world coordinate.
			 * @return The sample value at the current coordinate.
			*/
			STPSample_t get(int, int, int);

			/**
			 * @brief Empty the content of the cache, size is not changed. This operation is not atomic.
			*/
			void clear() noexcept;

			/**
			 * @brief Retrieve the size of the cache.
			 * @return The size of the cache.
			*/
			unsigned long long capacity() const noexcept;

		};

		//The ascendant layer will be executed before this layer, like a singly linked list
		//usually there is only one ascendant, but if there is a merge point in the chain, there will be multiple,
		//depended on the actual implementation Basically it's {asc*, asc*...}
		const std::unique_ptr<STPLayer*[]> Ascendant;
		//The cache in this layer
		std::optional<STPLayerCache> Cache;

		/**
		 * @brief Generate a unique seed for this layer.
		 * @param global_seed The seed that is used to generate the entire world.
		 * @param salt A random number that is used to mix the global seed to generate layer seed.
		 * @return The layer seed.
		*/
		static STPSeed_t seedLayer(STPSeed_t, STPSeed_t) noexcept;

		/**
		 * @brief Sample the layer, given the world coordinate and return a sample point.
		 * The value of the sample point can be interpreted differently in different layer, for instance climate, temp or biome id
		 * @param x The x coordinate on the terrain
		 * @param y The y coordinate on the terrain. If the one wants to generate 2D biome map, the actual implementation can ignore this parameter
		 * @param z The z coordinate on the terrain
		 * @return Sample id or value, depended on the actual implementation.
		*/
		virtual STPSample_t sample(int, int, int) = 0;

	protected:

		/**
		 * @brief Generate a unique seed for a coordinate in this layer.
		 * @param x The x coordinate in world.
		 * @param z The z coordinate in world.
		 * @return The local seed associated with the world coordinate.
		*/
		STPSeed_t seedLocal(int, int) const noexcept;

		/**
		 * @brief Create a random number generator for the specified local seed.
		 * @param local_seed The local seed from which the generator is built on.
		 * @return The generator with the specified local seed. The same local seed will always give the same sequence of random number.
		*/
		STPLocalSampler createLocalSampler(STPSeed_t) const noexcept;

		/**
		 * @brief Create a random number generator for the specified world coordinate.
		 * @param x The x coordinate in the world.
		 * @param z The z coordinate in the world.
		 * @return A deterministic generator for the specified world coordinate.
		 * @see createLocalSampler(STPSeed_t)
		*/
		STPLocalSampler createLocalSampler(int, int) const noexcept;

		/**
		 * @brief Mix seed with a factor to achieve a degree of randomness to form a new seed. This function guarantees that if two same values are the same,
		 * the returning seed will always be the same
		 * @param s The seed
		 * @param fac The factor that is used to mix
		 * @return The mixed seed
		*/
		static STPSeed_t mixSeed(STPSeed_t, long long) noexcept;

	public:

		//A initialiser list of pointers to ascendant layer(s).
		typedef std::initializer_list<std::reference_wrapper<STPLayer>> STPAscendantInitialiser;

		//Salt is a random value used to mix the global seed to generate layer and local seed
		const STPSeed_t Salt;
		//Seed for each layer, the same layer under the same world seed and salt will always have the same layer seed
		const STPSeed_t LayerSeed;

		/**
		 * @brief Create a layer instance.
		 * @tparam Asc... A list of ascendant, could be only none, could be only one, could be more... Only STPLayer is accepted.
		 * @param cache_size Cache size for this layer, it should be in the power of 2. Or 0 size to disable caching.
		 * @param global_seed The global seed is the seed that used to generate the entire world, a.k.a., world seed.
		 * @param salt The salt is a random number that used to mix the global to generate local and layer seed, such that each layer should use
		 * different salt value.
		 * @param ascendant... The next executed layer. If more than one layer is provided, the layer is merging.
		 * Each ascendant should be dynamically allocated, memory will be freed when the layers are destroyed.
		*/
		STPLayer(size_t, STPSeed_t, STPSeed_t, STPAscendantInitialiser = {});

		STPLayer(const STPLayer&) = delete;

		STPLayer(STPLayer&&) = delete;

		STPLayer& operator=(const STPLayer&) = delete;

		STPLayer& operator=(STPLayer&&) = delete;

		virtual ~STPLayer() = default;

		/**
		 * @brief Query the cache size on this layer cache
		 * @return The size of the layer cache for this layer
		*/
		size_t cacheSize() const noexcept;

		/**
		 * @brief It will first read from the layer cache with the given world coordinate as key, if the cache exists, retrieve the value directly. Otherwise
		 * the layer sample function will be called. The result is then stored into the cache and returned.
		 * @param x The x coordinate on the terrain
		 * @param y The y coordinate on the terrain
		 * @param z The z coordinate on the terrain
		 * @return The cached value
		*/
		STPSample_t retrieve(int, int, int);

		/**
		 * @brief Get the parent layer with specified index.
		 * @param index The index of the ascendant to get.
		 * If not provided, it will default to 0, which is the first parent layer.
		 * @return The ascendant at that index - the parent layers, who will be executed before this layer.
		 * There might be more than one ascendant in case there is a merge in the execution chain.
		 * The behaviour of out of bound index or no ascendant is undefined.
		*/
		STPLayer& getAscendant(size_t = 0u) noexcept;

		/**
		 * @brief Test if there is more than one parent in this layer
		 * @return True if yes
		*/
		bool isMerging() const noexcept;

	};
}
#endif//_STP_LAYER_H_