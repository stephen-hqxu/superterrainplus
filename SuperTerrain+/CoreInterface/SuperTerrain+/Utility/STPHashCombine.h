#pragma once
#ifndef _STP_HASH_COMBINE_H_
#define _STP_HASH_COMBINE_H_

//System
#include <type_traits>

namespace SuperTerrainPlus {

	/**
	 * @brief STPHashCombine contains a simple function for combining multiple hash results together
	*/
	class STPHashCombine {
	private:

		//static-only class, no instantiation is allowed
		STPHashCombine() = delete;

		~STPHashCombine() = delete;

		//TODO: change to template lambda and put it inside combine()

		/**
		 * @brief Combine a hash with the next value
		 * @tparam T The type of the next value
		 * @param seed The old hash input, and the output of combined hash
		 * @param value The next value to be hashed
		*/
		template<typename T>
		static void combineOne(size_t&, const T&);

	public:

		//Forwarding is not necessary because only primitive types can be hashed

		/**
		 * @brief Combine a hash with all values in the order specified
		 * @tparam ...T All types of values
		 * @param seed The old hash input
		 * @param value... All values that will be hashed and combined
		 * @return The resultant hash output that has been combined with all values
		*/
		template<typename... T>
		static size_t combine(size_t, const T&...);

	};

}
#include "STPHashCombine.inl"
#endif//_STP_HASH_COMBINE_H_