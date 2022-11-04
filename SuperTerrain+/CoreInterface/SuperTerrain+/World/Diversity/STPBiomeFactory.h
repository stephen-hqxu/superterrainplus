#pragma once
#ifndef _STP_BIOME_FACTORY_H_
#define _STP_BIOME_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
#include "STPLayer.h"
//Memory Management
#include "../../Utility/Memory/STPObjectPool.h"

//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPBiomeFactory provides a safe environment for multi-threaded biome map generation.
	*/
	class STP_API STPBiomeFactory {
	private:

		/**
		 * @brief STPProductionLineCreator creates a production line from the biome layer implementation.
		*/
		struct STPProductionLineCreator {
		private:

			STPBiomeFactory& Factory;

		public:

			/**
			 * @brief Initialise a production line creator.
			 * @param factory The dependent biome factory.
			*/
			STPProductionLineCreator(STPBiomeFactory&);

			STPLayer* operator()();

		};
		//Basically it behaves like a memory pool.
		//Whenever operator() is called, we search for an empty production line, and use that to generate biome.
		//If no available production line can be found, ask more production line from the manufacturer.
		STPObjectPool<STPLayer*, STPProductionLineCreator> LayerProductionLine;

		/**
		 * @brief A layer supplier, which provides the algorithm for layer generation and creates a new layer tree structure.
		 * This function is thread-safe and the call is guarded by a lock.
		 * @return A pointer to the root of the new layer tree structure.
		 * This should be pointing to the first layer in the layer tree structure,
		 * such that all layers can be traversed by visiting all successive the ascendant recursively.
		 * The memory of the entire layer tree should be managed by the application
		 * and must guarantee its lifetime should outlive the lifetime of the biome factory.
		*/
		virtual STPLayer* supply() = 0;

	public:

		//Specify the dimension of the generated biome map
		const glm::uvec2 BiomeDimension;

		/**
		 * @brief Init biome factory with internal cache memory pool that can be used for multi-threading, each thread will be automatically allocated one cache
		 * @param dimension The dimension of the biome map, this will init a 2D biome map generator, with x and z component only
		*/
		STPBiomeFactory(glm::uvec2);

		STPBiomeFactory(const STPBiomeFactory&) = delete;

		STPBiomeFactory(STPBiomeFactory&&) = delete;

		STPBiomeFactory& operator=(const STPBiomeFactory&) = delete;

		STPBiomeFactory& operator=(STPBiomeFactory&&) = delete;

		//make sure the thread stopped before deleting factory
		//stop all waiting workers and waiting for current worker to finish.
		virtual ~STPBiomeFactory() = default;

		/**
		 * @brief Generate a biome map using the biome layer implementation
		 * @param biomemap The output where biome map will be stored, must be preallocated with enough space
		 * @param offset The offset of the biome map, that is equivalent to the world coordinate.
		*/
		void operator()(Sample*, glm::ivec2);

	};

}
#endif//_STP_BIOME_FACTORY_H_