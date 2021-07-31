#pragma once
#ifndef _STP_BIOME_PROPERTY_HPP_
#define _STP_BIOME_PROPERTY_HPP_

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {

	/**
	 * @brief STPBiomeProperty contains parameters for generating a multi-biome terrain
	*/
	struct STPBiomeProperty {
	public:

		//Generation control
		//Determine the zooming of the noise map
		float Scale;
		//Control how many heightmap will be conbined
		unsigned int Octave;
		//Control how the amplitude will be changed in each octave. Range (0,1)
		float Persistence;
		//Control how the frequency will be changed in each octave.
		float Lacunarity;
		//The base height of the biome
		float Depth;
		//The variation from the base height of the biome
		float Variation;

	};

}
#endif//_STP_BIOME_PROPERTY_HPP_