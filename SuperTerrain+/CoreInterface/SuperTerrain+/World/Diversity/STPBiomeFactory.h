#pragma once
#ifndef _STP_BIOME_FACTORY_H_
#define _STP_BIOME_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//System
#include <queue>
#include <mutex>
#include <list>
//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
//Biome
#include "STPLayerManager.h"

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPBiomeFactory provides a safe environment for multi-threaded biome map generation.
	*/
	class STP_API STPBiomeFactory {
	private:

		typedef std::unique_ptr<STPLayerManager> STPLayerManager_t;
		//Basically it behaves like a memory pool.
		//Whenever operator() is called, we search for an empty production line, and use that to generate biome.
		//If no available production line can be found, ask more production line from the manufacturer.
		std::queue<STPLayerManager_t, std::list<STPLayerManager_t>> LayerProductionLine;
		mutable std::mutex ProductionLock;

		/**
		 * @brief A layer supplier, which provides the algorithm for layer chain generation.
		 * @return A new layer production line instance.
		*/
		virtual STPLayerManager supply() const = 0;

		/**
		 * @brief Request a production line.
		 * If no available production line is presented, a new production line is asked from layer supplier, and returned.
		 * @return Requested production line.
		*/
		STPLayerManager_t requestProductionLine();

		/**
		 * @brief Return the production back to idling queue.
		 * Any attemp to use the production line after returned will result in undefined behaviour
		 * @param line Production line to be returned
		*/
		void returnProductionLine(STPLayerManager_t&);

	protected:

		/**
		 * @brief Init biome factory with internal cache memory pool that can be used for multi-threading, each thread will be automatically allocaed one cache
		 * @param dimension The dimension of the biome map
		 * If the y component of the dimension is one, a 2D biome map will be generated
		*/
		STPBiomeFactory(glm::uvec3);

		/**
		 * @brief Init biome factory with internal cache memory pool that can be used for multi-threading, each thread will be automatically allocaed one cache
		 * @param dimension The dimension of the biome map, this will init a 2D biome map generator, with x and z component only
		*/
		STPBiomeFactory(glm::uvec2);

	public:

		STPBiomeFactory(const STPBiomeFactory&) = delete;

		STPBiomeFactory(STPBiomeFactory&&) = delete;

		STPBiomeFactory& operator=(const STPBiomeFactory&) = delete;

		STPBiomeFactory& operator=(STPBiomeFactory&&) = delete;

		//Specify the dimension of the generated biome map, in 3 dimension
		const glm::uvec3 BiomeDimension;

		//make sure the thread stopped before deleting factory
		//stop all waiting workers and waiting for current worker to finish.
		virtual ~STPBiomeFactory() = default;

		/**
		 * @brief Generate a biome map using the biome chain implementation
		 * @param biomemap The output where biome map will be stored, must be preallocated with enough space
		 * @param offset The offset of the biome map, that is equavalent to the world coordinate.
		*/
		void operator()(Sample*, glm::ivec3);

	};

}
#endif//_STP_BIOME_FACTORY_H_