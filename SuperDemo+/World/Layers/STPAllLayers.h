#pragma once
#ifndef _STP_ALL_LAYERS_H_
#define _STP_ALL_LAYERS_H_

#include <SuperTerrain+/World/Diversity/STPBiomeFactory.h>

//Storage
#include <list>

namespace STPDemo {

	/**
	 * @brief STPLayerChainBuilder is an example biome layer chain builder
	*/
	class STPLayerChainBuilder : public SuperTerrainPlus::STPDiversity::STPBiomeFactory {
	private:

		/**
		 * @brief STPLayerPipeline contains a complete pipeline of layers to generate a biome map.
		*/
		struct STPLayerPipeline;
		//store all constructed biome layer tree
		std::list<STPLayerPipeline> LayerStructureStorage;

		SuperTerrainPlus::STPDiversity::STPLayer& supply() override;

	public:

		const SuperTerrainPlus::STPSeed_t GlobalSeed;

		/**
		 * @brief Init the chain build
		 * @param dimension The biome map dimension in 2D
		 * @param global The global seed for generation
		*/
		STPLayerChainBuilder(glm::uvec2, SuperTerrainPlus::STPSeed_t);

		~STPLayerChainBuilder();

	};

}
#endif//_STP_ALL_LAYERS_H_