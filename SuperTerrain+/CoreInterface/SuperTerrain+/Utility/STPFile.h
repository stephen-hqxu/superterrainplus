#pragma once
#ifndef _STP_FILE_H_
#define _STP_FILE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <tuple>
#include <array>

#include <string>
#include <utility>

namespace SuperTerrainPlus {

	/**
	 * @brief STPFile is a handy file IO utility.
	*/
	namespace STPFile {

		/**
		 * @brief Inline implementation for template file functions.
		*/
		namespace STPFileImpl {

			//TODO: C++20 template lambda
			/**
			 * @brief Build a filename by concatenating path, file and extension.
			 * @tparam T The type of the index sequence.
			 * @param path The path to be prepended.
			 * @param file The name of the file at the middle.
			 * @param extension The extension to be appended.
			 * @param Index sequence for the path.
			 * @param Index sequence for the file.
			 * @param Index sequence for the extension.
			 * @return An array of concatenated string.
			*/
			template<size_t... IP, size_t... IF, size_t... IE, typename T>
			constexpr auto buildFilename(const char*, const char*, const char*,
				std::integer_sequence<T, IP...>, std::integer_sequence<T, IF...>, std::integer_sequence<T, IE...>);

		}

		/**
		 * @brief Open a file and read all contents in the file.
		 * @param filename The filename of the file to be read.
		 * @return A string containing all text contents of the file.
		*/
		STP_API std::string read(const char*);

		/**
		 * @brief Generate some filenames.
		 * @param path The path to the file to be prepended to each filename.
		 * @param file The name of the file.
		 * @param extension An array of extension to be appended one by one.
		 * @return If all extensions have the same length, returns an array of full filename.
		 * Else, it returns a tuple of full filename with different extensions.
		*/
		template<size_t LP, size_t LF, size_t... LE>
		constexpr auto generateFilename(const char(&)[LP], const char(&)[LF], const char(&...le)[LE]);

	}

}
#include "STPFile.inl"
#endif//_STP_FILE_H_