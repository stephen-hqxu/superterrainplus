#pragma once
#ifndef _STP_ALL_LAYERS_H_
#define _STP_ALL_LAYERS_H_

#include <SuperTerrain+/World/Diversity/STPBiomeFactory.h>

//Storage
#include <list>
#include <vector>
#include <memory>

namespace STPDemo {

	/**
	 * @brief STPLayerChainBuilder is an example biome layer chain builder
	*/
	class STPLayerChainBuilder : public SuperTerrainPlus::STPDiversity::STPBiomeFactory {
	private:

		//store all constructed biome layer tree
		std::list<std::vector<std::unique_ptr<SuperTerrainPlus::STPDiversity::STPLayer>>> LayerStructureStorage;

		SuperTerrainPlus::STPDiversity::STPLayer* supply() override;

	public:

		const SuperTerrainPlus::STPDiversity::Seed GlobalSeed;

		/**
		 * @brief Init the chain build
		 * @param dimension The biome map dimension in 2D
		 * @param global The global seed for generation
		*/
		STPLayerChainBuilder(glm::uvec2, SuperTerrainPlus::STPDiversity::Seed);

		~STPLayerChainBuilder() = default;

	};

}
#endif//_STP_ALL_LAYERS_H_