#pragma once
#ifndef _STP_HASH_COMBINE_H_
#define _STP_HASH_COMBINE_H_

//System
#include <type_traits>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPHashCombine contains a simple function for combining multiple hash results together
	*/
	class STPHashCombine {
	private:

		//static-only class, no instantiation is allowed
		STPHashCombine() = delete;

		~STPHashCombine() = delete;

	public:

		//Forwarding is not necessary because only primitive types can be hashed, and primitive type is best suited for pass-by-value

		/**
		 * @brief Combine a hash with the next value
		 * @tparam T The type of the next value
		 * @param seed The old hash input, and the output of combined hash
		 * @param value The next value to be hashed
		*/
		template<typename T>
		static void combine(size_t&, T);

		/**
		 * @brief Combine a hash with all values in the order specified
		 * @tparam ...T All types of values
		 * @param seed The old hash input, and resultant hash output that has been combined with all values
		 * @param value... All values that will be hashed and combined
		*/
		template<typename... T>
		static void combineAll(size_t&, T...);

	};

}
#include "STPHashCombine.inl"
#endif//_STP_HASH_COMBINE_H_