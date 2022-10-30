#pragma once
#ifndef _STP_STRING_UTILITY_H_
#define _STP_STRING_UTILITY_H_

#include <array>
#include <tuple>

namespace SuperTerrainPlus {

	/**
	 * @brief STPStringUtility contains some useful functions for manipulating string in compile time.
	*/
	namespace STPStringUtility {

		/**
		 * @brief Join a number of string literals. Assuming all of them are null-terminated.
		 * The last character of the every string is removed because that's a null.
		 * The concatenated string is null terminated.
		 * @param str... A variable number of string literals to be concatenated.
		 * @return The null-terminated joint string, represented in an std::array of char.
		*/
		template<size_t... L>
		[[nodiscard]] constexpr auto concatCharArray(const char(&... str)[L]) noexcept;

		/**
		 * @brief Generate an array of filename with different extensions.
		 * @param path The path to the file to be prepended to each file.
		 * @param file The name of the file.
		 * @param extension An array of extension to be appended one by one to each filename.
		 * @return If all extensions have the same length, returns an array of full filename.
		 * Else, it returns a tuple of full filename with different extensions.
		*/
		template<size_t LP, size_t LF, size_t... LE>
		[[nodiscard]] constexpr auto generateFilename(const char(&)[LP], const char(&)[LF], const char(&... le)[LE]) noexcept;

	}

}
#include "STPStringUtility.inl"
#endif//_STP_STRING_UTILITY_H_