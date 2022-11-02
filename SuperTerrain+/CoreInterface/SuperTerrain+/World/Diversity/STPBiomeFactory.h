#pragma once
#ifndef _STP_BIOME_FACTORY_H_
#define _STP_BIOME_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Biome
#include "STPLayerTree.h"
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

		typedef std::unique_ptr<STPLayerTree> STPLayerTree_t;

		/**
		 * @brief STPProductionLineCreator creates a production line from the supply chain.
		*/
		struct STPProductionLineCreator {
		private:

			const STPBiomeFactory& Factory;

		public:

			/**
			 * @brief Initialise a production line creator.
			 * @param factory The dependent biome factory.
			*/
			STPProductionLineCreator(const STPBiomeFactory&);

			STPLayerTree_t operator()() const;

		};
		//Basically it behaves like a memory pool.
		//Whenever operator() is called, we search for an empty production line, and use that to generate biome.
		//If no available production line can be found, ask more production line from the manufacturer.
		STPObjectPool<STPLayerTree_t, STPProductionLineCreator> LayerProductionLine;

		/**
		 * @brief A layer supplier, which provides the algorithm for layer chain generation.
		 * @return A new layer production line instance.
		*/
		virtual STPLayerTree supply() const = 0;

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
		 * @brief Generate a biome map using the biome chain implementation
		 * @param biomemap The output where biome map will be stored, must be preallocated with enough space
		 * @param offset The offset of the biome map, that is equivalent to the world coordinate.
		*/
		void operator()(Sample*, glm::ivec2);

	};

}
#endif//_STP_BIOME_FACTORY_H_