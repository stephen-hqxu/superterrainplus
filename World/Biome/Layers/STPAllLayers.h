#pragma once
#ifndef _STP_LAYERS_ALL_HPP_
#define _STP_LAYERS_ALL_HPP_

#include "../STPBiomeFactory.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;

	/**
	 * @brief STPLayerChainBuilder is an example biome layer chain builder
	*/
	class STPLayerChainBuilder : public SuperTerrainPlus::STPDiversity::STPBiomeFactory {
	private:

		SuperTerrainPlus::STPDiversity::STPLayerManager* supply() const override;

	public:

		const Seed GlobalSeed;

		/**
		 * @brief Init the chain build
		 * @param dimension The biome map dimension in 2D
		 * @param global The global seed for generation
		*/
		STPLayerChainBuilder(glm::uvec2, Seed);

		~STPLayerChainBuilder() = default;

	};

}
#endif//_STP_LAYERS_ALL_HPP_
