#pragma once
#ifndef _STP_INI_READER_H_
#define _STP_INI_READER_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>

#include "STPINIStorage.hpp"

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINIReader is an INI deserialiser which loads INI settings from a string and store the result in a container.
	*/
	class STP_ALGORITHM_HOST_API STPINIReader {
	private:

		//The implementation of the INI reader.
		class STPINIReaderImpl;

		//Contains parsed INI data
		STPINIStorage Data;

		/**
		 * @brief Emplace a new section into the storage.
		 * @param sec_name The name of the new section.
		 * @return The pointer to the new section.
		 * If the section name is duplicate, the section with given name will be returned.
		*/
		STPINISection& addSection(const std::string&);

	public:

		/**
		 * @brief Initialise an INI reader and start parsing the INI.
		 * @param ini_str A string contains all content of an INI.
		 * No reference is retained after this function returns.
		 * Any syntactic error in the INI string will halt the parser and causes exception to be thrown.
		*/
		STPINIReader(const std::string&);

		STPINIReader(const STPINIReader&) = default;

		STPINIReader(STPINIReader&&) noexcept = default;

		STPINIReader& operator=(const STPINIReader&) = default;

		STPINIReader& operator=(STPINIReader&&) noexcept = default;

		~STPINIReader() = default;

		/**
		 * @brief Get INI data parsed.
		 * @return A pointer to an INI storage instance.
		*/
		const STPINIStorage& operator*() const;

	};

}
#endif//_STP_INI_READER_H_