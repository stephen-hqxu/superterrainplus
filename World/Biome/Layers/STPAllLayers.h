#pragma once
#ifndef _STP_LAYERS_ALL_HPP_
#define _STP_LAYERS_ALL_HPP_

#include "../STPLayerManager.h"

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {
	using SuperTerrainPlus::STPDiversity::Seed;

	/**
	 * @brief STPLayerChainBuilder is an example biome layer chain builder
	*/
	class STPLayerChainBuilder {
	private:

		const Seed GlobalSeed;

	public:

		/**
		 * @brief Init the chain build
		 * @param global The global seed for generation
		*/
		STPLayerChainBuilder(Seed);

		~STPLayerChainBuilder() = default;

		/**
		 * @brief Build a biome layer chain
		 * @return A new biome layer chain
		*/
		SuperTerrainPlus::STPDiversity::STPLayerManager* operator()() const;

	};

}
#endif//_STP_LAYERS_ALL_HPP_
