#pragma once
#ifndef _STP_INI_PARSER_H_
#define _STP_INI_PARSER_H_

#include <SuperAlgorithm+Host/STPAlgorithmDefine.h>
#include "STPINIData.hpp"

#include <vector>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINIParser is a collection of utility to serialise and de-serialise between INI settings and INI data structure.
	*/
	namespace STPINIParser {

		/**
		 * @brief STPINIReaderResult contains output from parsing INI string.
		 * The parsed INI data are all view, which is non-owning and all memory depends on the input string.
		*/
		struct STPINIReaderResult {
		public:

			//The INI storage containing the sections and properties from the INI string.
			STPINIData::STPINIStorageView Storage;
			//The order of each INI section.
			STPINIData::STPINIEntryIndexView SectionOrder;
			//The order of property in each INI section, whose indices are given by the section order.
			std::vector<STPINIData::STPINIEntryIndexView> PropertyOrder;

		};

		/**
		 * @brief Convert INI string to INI data structure that can be read by the application.
		 * @param ini The A null-terminated string contains all content of an INI.
		 * @param ini_name The name of the INI string, for debugging purposes.
		 * @return The parsed output
		*/
		STP_ALGORITHM_HOST_API STPINIReaderResult read(const std::string_view&, const std::string_view&);

	}

}
#endif//_STP_INI_PARSER_H_