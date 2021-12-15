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
	class STP_API STPFile {
	private:

		std::string Content;

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
		constexpr static auto buildFilename(const char*, const char*, const char*,
			std::integer_sequence<T, IP...>, std::integer_sequence<T, IF...>, std::integer_sequence<T, IE...>);

	public:

		/**
		 * @brief Open a file and read all contents in the file.
		 * @param filename The filename to be opened.
		*/
		STPFile(const char*);

		STPFile(const STPFile&) = default;

		STPFile(STPFile&&) noexcept = default;

		STPFile& operator=(const STPFile&) = default;

		STPFile& operator=(STPFile&&) noexcept = default;

		~STPFile() = default;

		/**
		 * @brief Get the whole content with the file being read.
		 * @return The pointer to the string to the content of the file.
		*/
		const std::string& operator*() const;
		
		/**
		 * @brief Generate some filenames.
		 * @param path The path to the file to be prepeneded to each filename.
		 * @param file The name of the file.
		 * @param extension An array of extension to be appended one by one.
		 * @return If all extensions have the same length, returns an array of full filename.
		 * Else, it returns a tuple of full filename with different extensions.
		*/
		template<size_t LP, size_t LF, size_t... LE>
		constexpr static auto generateFilename(const char(&)[LP], const char(&)[LF], const char(&...le)[LE]);

	};

}
#include "STPFile.inl"
#endif//_STP_FILE_H_