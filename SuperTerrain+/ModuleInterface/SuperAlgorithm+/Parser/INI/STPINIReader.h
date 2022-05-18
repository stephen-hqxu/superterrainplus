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
		STPINIStorageView Data;

		/**
		 * @brief Emplace a new section into the storage.
		 * @param sec_name The name of the new section.
		 * @return The pointer to the new section.
		 * If the section name is duplicate, the section with given name will be returned.
		*/
		STPINISectionView& addSection(const std::string_view&);

	public:

		/**
		 * @brief Initialise an INI reader and start parsing the INI.
		 * @param ini_str A non-owning string contains all content of an INI.
		 * No reference is retained after this function returns, however the memory where the string is stored should be managed by the user,
		 * until the current instance and returned INI storage view is destroyed.
		 * Any syntactic error in the INI string will halt the parser and causes exception to be thrown.
		*/
		STPINIReader(const std::string_view&);

		STPINIReader(const STPINIReader&) = default;

		STPINIReader(STPINIReader&&) noexcept = default;

		STPINIReader& operator=(const STPINIReader&) = default;

		STPINIReader& operator=(STPINIReader&&) noexcept = default;

		~STPINIReader() = default;

		/**
		 * @brief Get INI data structure parsed.
		 * @return A pointer to an INI storage instance.
		 * The returned storage is only a view to the raw INI string provided by the user at initialisation.
		*/
		const STPINIStorageView& operator*() const;

	};

}
#endif//_STP_INI_READER_H_