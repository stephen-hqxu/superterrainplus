#pragma once
#ifndef _STP_LAYER_H_
#define _STP_LAYER_H_

//System
#include <vector>
#include <memory>
#include <type_traits>
//RNG
#include "STPSeedMixer.h"
//Biome define
#include "STPLayerCache.h"

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
		 * @brief Super Terrain Plus uses layered biome generation algorithm (ref: Minecraft),
		 * STPLayer provides a abstract base class for implementations of subsequent layers.
		*/
		class STPLayer {
		public:

			friend class STPLayerManager;

			/**
			 * @brief STPLocalRNG is a random number generator for each local seed (a seed that is deterministic on world coordinate)
			*/
			struct STPLocalRNG final {
			protected:

				friend class STPLayer;

			private:

				//local seed varies from world coordinate, but the same coord will always yeild the same local seed
				const Seed LocalSeed;

				/**
				 * @brief Init rng for each local coordinate
				 * @param local_seed Local seed that is unique to each world coordinate
				*/
				STPLocalRNG(Seed);

			public:

				~STPLocalRNG();

				/**
				 * @brief Get the next random number in the sequence
				 * @param range The range from 0u to the specified value
				 * @return The next sequence
				*/
				Sample nextVal(Sample) const;

				/**
				 * @brief Select randomly among two variables
				 * @param var1 The first variable to choose
				 * @param var2 The second variable to choose
				 * @return The chosen value of the variable
				*/
				Sample choose(Sample, Sample) const;

				/**
				 * @brief Select randomly among four variables
				 * @param var1 The first variable to choose
				 * @param vaw2 The second variable to choose
				 * @param var3 The third variable to choose
				 * @param vaw4 The forth variable to choose
				 * @return The chosen value of the variable
				*/
				Sample choose(Sample, Sample, Sample, Sample) const;

			};

		private:

			//The ascendant layer will be executed before this layer, like a singly linked list
			//usually there is only one ascendant, but if there is a merge point in the chain, there will be multiple, depended on the actual implementation
			//Basically it's {asc*, asc*...}
			//TODO: Use shared_ptr instead
			std::vector<STPLayer*> Ascendant;

			//layer cache for dynamic programming
			std::unique_ptr<STPLayerCache> Cache;

			/**
			 * @brief Generate a unique seed for this layer
			 * @param global_seed The seed that is used to generate the entire world
			 * @param salt A random number that is used to mix the global seed to generate layer seed
			 * @return The layer seed
			*/
			static Seed genLayerSeed(Seed, Seed);

		protected:

			/**
			 * @brief Create a layer instance
			 * @tparam Asc A list of ascendants, could be only none, could be only one, could be more... Only STPLayer is accepted
			 * @param global_seed The global seed is the seed that used to generate the entire world, a.k.a., world seed.
			 * @param salt The salt is a random number that used to mix the global to generate local and layer seed, such that each layer should use
			 * different salt value
			 * @param ascendant The next executed layer. If more than one layer is provided, the layer is merging.
			 * Each ascendant should be dynamically allocated, memory will be freed when the layers are destroied.
			*/
			template <class... Asc>
			STPLayer(Seed, Seed, Asc*...);

			//Layer deletion is handled by STPLayerManager
			virtual ~STPLayer() = default;

			/**
			 * @brief Generate a unique seed for this coordinate in this layer
			 * @param x The x coordinate in world
			 * @param z The z coordinate in world
			 * @return The local seed associated with the world coordinate
			*/
			Seed genLocalSeed(int, int) const;

			/**
			 * @brief Get the random number generator for the specified local seed
			 * @param local_seed The local seed from which the RNG is built on.
			 * @return The generator with the specified local seed. The same local seed will always give the same sequence of random number
			*/
			STPLocalRNG getRNG(Seed) const;

		public:

			//Salt is a random value used to mix the global seed to generate layer and local seed
			const Seed Salt;
			//Seed for each layer, the same layer under the same world seed and salt will always have the same layer seed
			const Seed LayerSeed;

			STPLayer(const STPLayer&) = delete;

			STPLayer(STPLayer&&) = delete;

			STPLayer& operator=(const STPLayer&) = delete;

			STPLayer& operator=(STPLayer&&) = delete;

			/**
			 * @brief Query the cache size on this layer cache
			 * @return The size of the layer cache for this layer
			*/
			size_t cacheSize() const;

			/**
			 * @brief Sample the layer, given the world coordinate and return a sample point.
			 * The value of the sample point can be interpreted differently in different layer, for instance climate, temp or biome id
			 * @param x The x coordinate on the terrain
			 * @param y The y coordinate on the terrain. If the one wants to generate 2D biome map, the actual implementation can ignore this parameter
			 * @param z The z coordinate on the terrain
			 * @return Sample id or value, depended on the actual implementation.
			*/
			virtual Sample sample(int, int, int) = 0;

			/**
			 * @brief @link sample()
			 * It will first read from the layer cache with the given world coordinate as key, if the cache exists, return the value directly, otherwise
			 * the layer sample function will be called. The result is then stored into the cache and returned.
			 * @param x The x coordinate on the terrain
			 * @param y The y coordinate on the terrain
			 * @param z The z coordinate on the terrain
			 * @return The cached value
			*/
			Sample sample_cached(int, int, int);

			/**
			 * @brief Get the parent layer with specified index
			 * @param index The index of the ascendant to get
			 * @return The ascendants at that index - the parent layers, who will be executed before this layer.
			 * Return null if index out of bound or no ascendant
			*/
			STPLayer* const getAscendant(unsigned int) const;

			/**
			 * @brief Get the first parent layer
			 * @return The ascendants - the parent layers, who will be executed before this layer.
			 * There might be more than one ascendant in case there is a merge in the execution chain.
			 * Return null if there is no ascendant
			*/
			STPLayer* const getAscendant() const;

			/**
			 * @brief Get the number of ascendant in this layer, if there are more than one, it's a merging layer
			 * @return The number of ascendant in this layer
			*/
			size_t getAscendantCount() const;


			/**
			 * @brief Test if there is more than one parent in this layer
			 * @return True if yes
			*/
			bool isMerging() const;

			/**
			 * @brief Test if the current layer has any parent
			 * @return True if any
			*/
			bool hasAscendant() const;

		};
	}
}
#include "STPLayer.inl"
#endif//_STP_LAYER_H_